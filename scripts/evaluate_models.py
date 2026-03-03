#!/usr/bin/env python3
"""
Model Evaluation Script

Evaluates trained models (Qdrant hybrid search + XGBoost) on test data.
Generates detailed performance metrics, confusion matrix, and per-category analysis.

Usage:
    python scripts/evaluate_models.py --version v1.0
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support
)
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Load config
def load_config():
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("config/config.yaml not found")
    
    with open(config_path) as f:
        return yaml.safe_load(f)

# Load test data
def load_test_data(version: str) -> pd.DataFrame:
    """Load synthetic transaction data for evaluation."""
    data_path = Path(f"data/versions/{version}/synthetic_data.csv")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Test data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Use last 20% as test set (assuming 80/20 split during training)
    test_size = int(len(df) * 0.2)
    df_test = df.tail(test_size).reset_index(drop=True)
    
    print(f"Loaded {len(df_test)} test transactions")
    
    return df_test

# Load trained models
def load_trained_models(version: str):
    """Load XGBoost model, label encoder, and feature names."""
    
    print(f"Loading trained models for version: {version}")
    
    # Load XGBoost model
    model_path = Path(f"data/versions/{version}/xgboost_model.json")
    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost model not found: {model_path}")
    
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(model_path))
    print(f"Loaded XGBoost model from: {model_path}")
    
    # Load label encoder
    label_encoder_path = Path(f"data/versions/{version}/label_encoder.json")
    with open(label_encoder_path) as f:
        label_mapping = json.load(f)
    
    # Create reverse mapping (category -> index)
    label_to_idx = {v: int(k) for k, v in label_mapping.items()}
    idx_to_label = {int(k): v for k, v in label_mapping.items()}
    
    print(f"Loaded label encoder with {len(label_mapping)} categories")
    
    # Load feature names
    feature_names_path = Path(f"data/versions/{version}/feature_names.json")
    with open(feature_names_path) as f:
        feature_names = json.load(f)
    
    print(f"Loaded {len(feature_names)} feature names")
    
    return xgb_model, label_to_idx, idx_to_label, feature_names

# Load Qdrant index
def load_qdrant_index(version: str):
    """Load Qdrant client and collection."""
    
    storage_path = Path(f"data/versions/{version}/hybrid_index")
    if not storage_path.exists():
        raise FileNotFoundError(f"Qdrant index not found: {storage_path}")
    
    client = QdrantClient(path=str(storage_path))
    collection_name = f"transactions_{version}"
    
    print(f"Loaded Qdrant index from: {storage_path}")
    
    return client, collection_name

# Initialise embedding model
def init_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Initialise sentence transformer model."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    return model

# Preprocess text
def preprocess_description(text: str) -> str:
    """Clean and normalise transaction description."""
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text

# Generate embeddings
def generate_embeddings(df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    """Generate embeddings for test data."""
    print("Generating embeddings for test data...")
    descriptions = df['description'].apply(preprocess_description).tolist()
    embeddings = model.encode(descriptions, show_progress_bar=True, batch_size=256)
    return embeddings

# Extract features
def extract_features(
    description: str,
    amount: float,
    transaction_type: str,
    date: str,
    hybrid_results: List[Dict] = None
) -> Dict:
    """Extract features for XGBoost classification."""
    
    features = {}
    
    # Hybrid search features
    if hybrid_results and len(hybrid_results) > 0:
        features['top1_score'] = hybrid_results[0]['score'] if len(hybrid_results) > 0 else 0.0
        features['top2_score'] = hybrid_results[1]['score'] if len(hybrid_results) > 1 else 0.0
        features['top3_score'] = hybrid_results[2]['score'] if len(hybrid_results) > 2 else 0.0
        features['score_gap'] = features['top1_score'] - features['top2_score']
    else:
        features['top1_score'] = 0.0
        features['top2_score'] = 0.0
        features['top3_score'] = 0.0
        features['score_gap'] = 0.0
    
    # Amount features
    features['amount_log'] = np.log1p(abs(amount))
    features['is_round_number'] = 1 if amount % 100 == 0 else 0
    
    # Amount buckets
    if amount < 100:
        features['amount_bucket'] = 0
    elif amount < 1000:
        features['amount_bucket'] = 1
    elif amount < 10000:
        features['amount_bucket'] = 2
    else:
        features['amount_bucket'] = 3
    
    # Transaction type
    features['is_debit'] = 1 if transaction_type == 'DEBIT' else 0
    
    # Description patterns
    desc_lower = description.lower()
    features['has_merchant'] = 1 if re.match(r'^[A-Z]{3,}', description) else 0
    features['has_ref_number'] = 1 if re.search(r'\d{6,}', description) else 0
    features['has_fp_bacs'] = 1 if ('faster payment' in desc_lower or 'bacs' in desc_lower) else 0
    features['word_count'] = len(description.split())
    
    # Temporal features
    try:
        dt = pd.to_datetime(date)
        features['day_of_month'] = dt.day
        features['is_month_end'] = 1 if dt.day >= 25 else 0
    except:
        features['day_of_month'] = 15
        features['is_month_end'] = 0
    
    # UK-specific patterns
    features['has_hmrc'] = 1 if 'hmrc' in desc_lower else 0
    features['has_vat'] = 1 if 'vat' in desc_lower else 0
    features['has_stripe'] = 1 if 'stripe' in desc_lower else 0
    
    return features

# Run predictions
def run_predictions(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    xgb_model: xgb.XGBClassifier,
    client: QdrantClient,
    collection_name: str,
    idx_to_label: Dict,
    feature_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Run predictions on test set."""
    
    print("\nRunning predictions on test set...")
    
    predictions = []
    confidences = []
    feature_list = []
    
    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"Predicting {idx}/{len(df)}...")
        
        # Query hybrid search
        results = client.query_points(
            collection_name=collection_name,
            query=embeddings[idx].tolist(),
            limit=3
        ).points
        
        hybrid_results = [
            {'score': r.score, 'category': r.payload['category']}
            for r in results
        ]
        
        # Extract features
        features = extract_features(
            description=row['description'],
            amount=row['amount'],
            transaction_type=row['transaction_type'],
            date=row['date'],
            hybrid_results=hybrid_results
        )
        
        # Add embedding features
        for i in range(min(50, embeddings.shape[1])):
            features[f'emb_{i}'] = embeddings[idx][i]
        
        feature_list.append(features)
    
    # Convert to feature matrix
    feature_df = pd.DataFrame(feature_list)
    
    # Ensure feature order matches training
    feature_df = feature_df[feature_names]
    
    X = feature_df.values
    
    # Predict
    y_pred_encoded = xgb_model.predict(X)
    y_pred_proba = xgb_model.predict_proba(X)
    
    # Convert to category labels
    y_pred = [idx_to_label[int(idx)] for idx in y_pred_encoded]
    confidences = np.max(y_pred_proba, axis=1)
    
    return np.array(y_pred), confidences

# Calculate metrics
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, confidences: np.ndarray):
    """Calculate comprehensive evaluation metrics."""
    
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Precision, Recall, F1 per category
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Weighted averages
    precision_weighted = np.average(precision, weights=support)
    recall_weighted = np.average(recall, weights=support)
    f1_weighted = np.average(f1, weights=support)
    
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Weighted Recall: {recall_weighted:.4f}")
    print(f"Weighted F1-Score: {f1_weighted:.4f}")
    
    # Confidence statistics
    print(f"\nConfidence Statistics:")
    print(f"  Mean confidence: {np.mean(confidences):.4f}")
    print(f"  Median confidence: {np.median(confidences):.4f}")
    print(f"  Min confidence: {np.min(confidences):.4f}")
    print(f"  Max confidence: {np.max(confidences):.4f}")
    
    # Confidence thresholds
    high_conf = np.sum(confidences >= 0.90)
    med_conf = np.sum((confidences >= 0.75) & (confidences < 0.90))
    low_conf = np.sum(confidences < 0.75)
    
    print(f"\nConfidence Distribution:")
    print(f"  High (>=0.90): {high_conf} ({high_conf/len(confidences)*100:.1f}%)")
    print(f"  Medium (0.75-0.90): {med_conf} ({med_conf/len(confidences)*100:.1f}%)")
    print(f"  Low (<0.75): {low_conf} ({low_conf/len(confidences)*100:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'mean_confidence': np.mean(confidences),
        'median_confidence': np.median(confidences),
    }

# Per-category analysis
def per_category_analysis(y_true: np.ndarray, y_pred: np.ndarray):
    """Analyse performance per category."""
    
    print("\n" + "=" * 60)
    print("PER-CATEGORY PERFORMANCE")
    print("=" * 60)
    
    # Get unique categories
    categories = sorted(set(y_true) | set(y_pred))
    
    # Calculate metrics per category
    results = []
    for cat in categories:
        mask_true = y_true == cat
        mask_pred = y_pred == cat
        
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(mask_true)
        
        results.append({
            'category': cat,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        })
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('f1', ascending=False)
    
    # Print top 10 and bottom 10
    print("\nTop 10 Categories (by F1-Score):")
    print(df_results.head(10).to_string(index=False))
    
    print("\nBottom 10 Categories (by F1-Score):")
    print(df_results.tail(10).to_string(index=False))
    
    return df_results

# Save evaluation results
def save_evaluation_results(
    version: str,
    metrics: Dict,
    df_category_results: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray
):
    """Save evaluation results to files."""
    
    print("\nSaving evaluation results...")
    
    # Create evaluation directory
    eval_dir = Path(f"data/versions/{version}/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    metrics_serializable = {
        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
        for k, v in metrics.items()
    }
    
    # Save overall metrics
    metrics_path = eval_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)  # Changed from metrics to metrics_serializable
    print(f"Saved metrics to: {metrics_path}")
    
    # Save per-category results
    category_results_path = eval_dir / "category_results.csv"
    df_category_results.to_csv(category_results_path, index=False)
    print(f"Saved category results to: {category_results_path}")
    
    # Save predictions
    predictions_path = eval_dir / "predictions.csv"
    df_predictions = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'confidence': confidences,
        'correct': y_true == y_pred
    })
    df_predictions.to_csv(predictions_path, index=False)
    print(f"Saved predictions to: {predictions_path}")
    
    # Save classification report
    report_path = eval_dir / "classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(classification_report(y_true, y_pred, zero_division=0))
    print(f"Saved classification report to: {report_path}")

# Main evaluation function
def evaluate_models(version: str, config: Dict):
    """Main evaluation pipeline."""
    
    print(f"\nStarting model evaluation for version: {version}")
    print("=" * 60)
    
    # Load test data
    df_test = load_test_data(version)
    
    # Load trained models
    xgb_model, label_to_idx, idx_to_label, feature_names = load_trained_models(version)
    
    # Load Qdrant index
    client, collection_name = load_qdrant_index(version)
    
    # Initialise embedding model
    embedding_model = init_embedding_model()
    
    # Generate embeddings for test data
    embeddings = generate_embeddings(df_test, embedding_model)
    
    # Run predictions
    y_pred, confidences = run_predictions(
        df=df_test,
        embeddings=embeddings,
        xgb_model=xgb_model,
        client=client,
        collection_name=collection_name,
        idx_to_label=idx_to_label,
        feature_names=feature_names
    )
    
    # Get ground truth
    y_true = df_test['category'].values
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, confidences)
    
    # Per-category analysis
    df_category_results = per_category_analysis(y_true, y_pred)
    
    # Save results
    save_evaluation_results(
        version=version,
        metrics=metrics,
        df_category_results=df_category_results,
        y_true=y_true,
        y_pred=y_pred,
        confidences=confidences
    )
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: data/versions/{version}/evaluation/")

# CLI
def main():
    parser = argparse.ArgumentParser(description="Evaluate transaction categorisation models")
    parser.add_argument("--version", default="v1.0", help="Model version (default: v1.0)")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    # Evaluate models
    evaluate_models(version=args.version, config=config)

if __name__ == "__main__":
    main()