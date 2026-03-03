"""
Core Schemas Module

Pydantic models for request/response validation used across API and batch processing.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal
from datetime import datetime
from decimal import Decimal


class TransactionInput(BaseModel):
    """Input schema for a single transaction to be categorised."""
    
    account_id: str = Field(..., description="Account identifier")
    transaction_id: str = Field(..., description="Unique transaction identifier")
    description: str = Field(..., description="Transaction description from bank feed")
    amount: float = Field(..., description="Transaction amount (positive number)")
    transaction_type: Literal["DEBIT", "CREDIT"] = Field(..., description="Transaction direction")
    date: str = Field(..., description="Transaction date in YYYY-MM-DD format")
    currency_code: str = Field(default="GBP", description="Currency code")
    
    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v):
        """Ensure amount is positive."""
        if v <= 0:
            raise ValueError("Amount must be positive")
        return v
    
    @field_validator('date')
    @classmethod
    def validate_date(cls, v):
        """Validate date format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        """Ensure description is not empty."""
        if not v or not v.strip():
            raise ValueError("Description cannot be empty")
        return v.strip()


class CategorisationResult(BaseModel):
    """Output schema for categorisation result."""
    
    account_id: str
    transaction_id: str
    description: str
    amount: float
    transaction_type: Literal["DEBIT", "CREDIT"]
    date: str
    currency_code: str
    
    # Categorisation results
    category: str = Field(..., description="Leaf category code")
    primary_category: str = Field(..., description="Top-level category code")
    full_category_path: str = Field(..., description="Full category path (e.g., 'REVENUE > Card Payments')")
    
    # Metadata
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)", ge=0.0, le=1.0)
    categorisation_method: Literal[
        "EXACT_MATCH",
        "FUZZY_MATCH", 
        "SEMANTIC_KEYWORD",
        "SEMANTIC_FULL",
        "HYBRID_XGB",
        "LLM_FALLBACK"
    ] = Field(
        ..., 
        description="Method used for categorisation"
    )
    model_version: str = Field(..., description="Model version used (e.g., 'v1.0')")
    
    class Config:
        json_schema_extra = {
            "example": {
                "account_id": "ACC123456",
                "transaction_id": "TXN7890123",
                "description": "STRIPE PAYMENT",
                "amount": 1250.50,
                "transaction_type": "CREDIT",
                "date": "2024-02-15",
                "currency_code": "GBP",
                "category": "card_payments",
                "primary_category": "REVENUE",
                "full_category_path": "Revenue > Card Payments",
                "confidence": 0.95,
                "categorisation_method": "HYBRID_XGB",
                "model_version": "v1.0"
            }
        }


class BatchCategorisationRequest(BaseModel):
    """Request schema for batch categorisation."""
    
    transactions: List[TransactionInput] = Field(
        ..., 
        description="List of transactions to categorise",
        min_length=1,
        max_length=10000
    )
    version: Optional[str] = Field(
        None, 
        description="Model version to use (uses default if not specified)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "transactions": [
                    {
                        "account_id": "ACC123456",
                        "transaction_id": "TXN001",
                        "description": "STRIPE PAYMENT",
                        "amount": 1250.50,
                        "transaction_type": "CREDIT",
                        "date": "2024-02-15",
                        "currency_code": "GBP"
                    }
                ],
                "version": "v1.0"
            }
        }


class BatchCategorisationResponse(BaseModel):
    """Response schema for batch categorisation."""
    
    results: List[CategorisationResult] = Field(..., description="Categorisation results")
    total_processed: int = Field(..., description="Total number of transactions processed")
    model_version: str = Field(..., description="Model version used")
    processing_time_seconds: float = Field(..., description="Total processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "account_id": "ACC123456",
                        "transaction_id": "TXN001",
                        "description": "STRIPE PAYMENT",
                        "amount": 1250.50,
                        "transaction_type": "CREDIT",
                        "date": "2024-02-15",
                        "currency_code": "GBP",
                        "category": "card_payments",
                        "primary_category": "REVENUE",
                        "full_category_path": "Revenue > Card Payments",
                        "confidence": 0.95,
                        "categorisation_method": "HYBRID_XGB",
                        "model_version": "v1.0"
                    }
                ],
                "total_processed": 1,
                "model_version": "v1.0",
                "processing_time_seconds": 0.123
            }
        }


class SingleCategorisationRequest(BaseModel):
    """Request schema for single transaction categorisation."""
    
    transaction: TransactionInput = Field(..., description="Transaction to categorise")
    version: Optional[str] = Field(
        None, 
        description="Model version to use (uses default if not specified)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction": {
                    "account_id": "ACC123456",
                    "transaction_id": "TXN001",
                    "description": "STRIPE PAYMENT",
                    "amount": 1250.50,
                    "transaction_type": "CREDIT",
                    "date": "2024-02-15",
                    "currency_code": "GBP"
                },
                "version": "v1.0"
            }
        }


class SingleCategorisationResponse(BaseModel):
    """Response schema for single transaction categorisation."""
    
    result: CategorisationResult = Field(..., description="Categorisation result")
    model_version: str = Field(..., description="Model version used")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "result": {
                    "account_id": "ACC123456",
                    "transaction_id": "TXN001",
                    "description": "STRIPE PAYMENT",
                    "amount": 1250.50,
                    "transaction_type": "CREDIT",
                    "date": "2024-02-15",
                    "currency_code": "GBP",
                    "category": "card_payments",
                    "primary_category": "REVENUE",
                    "full_category_path": "Revenue > Card Payments",
                    "confidence": 0.95,
                    "categorisation_method": "HYBRID_XGB",
                    "model_version": "v1.0"
                },
                "model_version": "v1.0",
                "processing_time_seconds": 0.045
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: Literal["healthy", "unhealthy"] = Field(..., description="Service health status")
    version: str = Field(..., description="Default model version")
    available_versions: List[str] = Field(..., description="List of available model versions")
    timestamp: str = Field(..., description="Current timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "v1.0",
                "available_versions": ["v1.0", "v2.0"],
                "timestamp": "2024-02-15T10:30:00Z"
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid transaction data",
                "detail": "Amount must be positive"
            }
        }