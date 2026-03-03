"""
FastAPI Application

REST API for transaction categorisation service.
Provides endpoints for single transaction and batch categorisation.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
from datetime import datetime
from typing import Optional

from categoriser.core.config import get_config
from categoriser.core.schemas import (
    TransactionInput,
    CategorisationResult,
    SingleCategorisationRequest,
    SingleCategorisationResponse,
    BatchCategorisationRequest,
    BatchCategorisationResponse,
    HealthResponse,
    ErrorResponse
)
from categoriser.engine.orchestrator import TransactionCategoriser

# Initialise FastAPI app
app = FastAPI(
    title="SME Transaction Categorisation API",
    description="AI-powered transaction categorisation for UK SME bank transactions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator cache (loaded on demand per version)
orchestrators = {}


def get_orchestrator(version: Optional[str] = None) -> TransactionCategoriser:
    """
    Get or create orchestrator for specified version.
    
    Args:
        version: Model version (uses default if not specified)
    
    Returns:
        TransactionCategoriser instance
    """
    config = get_config()
    
    # Use default version if not specified
    if version is None:
        version = config.default_version
    
    # Return cached orchestrator if available
    if version in orchestrators:
        return orchestrators[version]
    
    # Create and cache new orchestrator
    try:
        orchestrator = TransactionCategoriser(version=version)
        orchestrators[version] = orchestrator
        return orchestrator
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model version {version}: {str(e)}"
        )


def map_method_to_api_enum(tier: str, method: str) -> str:
    """
    Map internal tier/method to API categorisation method enum
    
    Args:
        tier: Internal tier (tier1, tier2, tier3)
        method: Internal method name
    
    Returns:
        API enum value (EXACT_MATCH, FUZZY_MATCH, SEMANTIC_KEYWORD, SEMANTIC_FULL, HYBRID_XGB)
    """
    if tier == 'tier1':
        return 'EXACT_MATCH' if 'exact' in method else 'FUZZY_MATCH'
    elif tier == 'tier2':
        if 'keyword_direct' in method:
            return 'SEMANTIC_KEYWORD'
        elif 'keyword' in method:
            return 'SEMANTIC_KEYWORD'
        else:
            return 'SEMANTIC_FULL'
    else:  # tier3
        return 'HYBRID_XGB'


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "service": "SME Transaction Categorisation API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "categorise": "/categorise",
            "batch": "/batch",
            "versions": "/versions",
            "stats": "/stats",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and available model versions.
    """
    try:
        config = get_config()
        available_versions = config.list_available_versions()
        
        return HealthResponse(
            status="healthy",
            version=config.default_version,
            available_versions=available_versions,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            version="unknown",
            available_versions=[],
            timestamp=datetime.utcnow().isoformat() + "Z"
        )


@app.post("/categorise", response_model=SingleCategorisationResponse)
async def categorise_transaction(request: SingleCategorisationRequest):
    """
    Categorise a single transaction.
    
    Args:
        request: SingleCategorisationRequest with transaction details
    
    Returns:
        SingleCategorisationResponse with categorisation result
    
    Raises:
        HTTPException: If categorisation fails
    """
    try:
        # Get orchestrator
        orchestrator = get_orchestrator(request.version)
        
        # Convert Pydantic model to dict
        transaction_dict = request.transaction.model_dump()
        
        # Categorise transaction
        start_time = time.time()
        categorisation_result = orchestrator.categorise_transaction(transaction_dict)
        processing_time = time.time() - start_time
        
        # Map tier/method to API enum
        tier = categorisation_result.get('tier', 'tier3')
        method = categorisation_result.get('method', 'tier3_xgboost')
        cat_method = map_method_to_api_enum(tier, method)
        
        # Extract category components
        category = categorisation_result['category']
        primary_category = category.split('_')[0] if category != 'UNKNOWN' else 'UNKNOWN'
        
        # Build result
        result = {
            **transaction_dict,  # Original transaction fields
            'category': category,
            'primary_category': primary_category,
            'full_category_path': category,
            'categorisation_method': cat_method,
            'confidence': categorisation_result['confidence'],
            'model_version': orchestrator.version,
        }
        
        return SingleCategorisationResponse(
            result=result,
            model_version=orchestrator.version,
            processing_time_seconds=processing_time
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Categorisation failed: {str(e)}"
        )


@app.post("/batch", response_model=BatchCategorisationResponse)
async def categorise_batch(request: BatchCategorisationRequest):
    """
    Categorise multiple transactions in batch.
    
    Args:
        request: BatchCategorisationRequest with list of transactions
    
    Returns:
        BatchCategorisationResponse with all categorisation results
    
    Raises:
        HTTPException: If batch categorisation fails
    """
    try:
        # Validate batch size
        if len(request.transactions) > 10000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch size exceeds maximum of 10,000 transactions"
            )
        
        # Get orchestrator
        orchestrator = get_orchestrator(request.version)
        
        # Convert Pydantic models to dicts
        transactions_dicts = [txn.model_dump() for txn in request.transactions]
        
        # Categorise batch
        start_time = time.time()
        categorisation_results = orchestrator.categorise_batch(transactions_dicts)
        processing_time = time.time() - start_time
        
        # Build results
        results = []
        for i, cat_result in enumerate(categorisation_results):
            tier = cat_result.get('tier', 'tier3')
            method = cat_result.get('method', 'tier3_xgboost')
            cat_method = map_method_to_api_enum(tier, method)
            
            # Extract category components
            category = cat_result['category']
            primary_category = category.split('_')[0] if category != 'UNKNOWN' else 'UNKNOWN'
            
            merged_result = {
                **transactions_dicts[i],  # Original transaction fields
                'category': category,
                'primary_category': primary_category,
                'full_category_path': category,
                'categorisation_method': cat_method,
                'confidence': cat_result['confidence'],
                'model_version': orchestrator.version,
            }
            results.append(merged_result)
        
        return BatchCategorisationResponse(
            results=results,
            total_processed=len(results),
            model_version=orchestrator.version,
            processing_time_seconds=processing_time
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch categorisation failed: {str(e)}"
        )


@app.get("/versions", response_model=dict)
async def list_versions():
    """
    List all available model versions.
    
    Returns:
        Dictionary with default version and available versions
    """
    try:
        config = get_config()
        available_versions = config.list_available_versions()
        
        return {
            "default_version": config.default_version,
            "available_versions": available_versions
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list versions: {str(e)}"
        )


@app.get("/stats", response_model=dict)
async def get_statistics(version: Optional[str] = None):
    """
    Get categorisation statistics for a specific model version.
    
    Args:
        version: Model version (uses default if not specified)
    
    Returns:
        Dictionary with categorisation statistics
    """
    try:
        orchestrator = get_orchestrator(version)
        
        stats = orchestrator.stats
        total = stats['total']
        
        if total == 0:
            return {
                "version": orchestrator.version,
                "total_categorised": 0,
                "tier_breakdown": {},
                "performance": {}
            }
        
        return {
            "version": orchestrator.version,
            "total_categorised": total,
            "tier_breakdown": {
                "tier1_exact": {
                    "count": stats['tier1_exact'],
                    "percentage": round(stats['tier1_exact'] / total * 100, 2)
                },
                "tier1_fuzzy": {
                    "count": stats['tier1_fuzzy'],
                    "percentage": round(stats['tier1_fuzzy'] / total * 100, 2)
                },
                "tier2_keyword_direct": {
                    "count": stats['tier2_keyword_direct'],
                    "percentage": round(stats['tier2_keyword_direct'] / total * 100, 2)
                },
                "tier2_keyword_semantic": {
                    "count": stats['tier2_keyword_semantic'],
                    "percentage": round(stats['tier2_keyword_semantic'] / total * 100, 2)
                },
                "tier2_full_semantic": {
                    "count": stats['tier2_full_semantic'],
                    "percentage": round(stats['tier2_full_semantic'] / total * 100, 2)
                },
                "tier3_xgboost": {
                    "count": stats['tier3_xgboost'],
                    "percentage": round(stats['tier3_xgboost'] / total * 100, 2)
                }
            },
            "performance": {
                "tier1_avg_ms": round(sum(stats['tier1_time_ms']) / len(stats['tier1_time_ms']), 2) if stats['tier1_time_ms'] else 0,
                "tier2_avg_ms": round(sum(stats['tier2_time_ms']) / len(stats['tier2_time_ms']), 2) if stats['tier2_time_ms'] else 0,
                "tier3_avg_ms": round(sum(stats['tier3_time_ms']) / len(stats['tier3_time_ms']), 2) if stats['tier3_time_ms'] else 0
            }
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="ValidationError",
            message="Invalid input data",
            detail=str(exc)
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            detail=str(exc)
        ).model_dump()
    )


# Run with: uvicorn categoriser.api.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)