#!/usr/bin/env python3
"""
Model Training Pipeline - SME Transaction Categorisation

Training approach:
1. Direction-filter category space at start (DEBIT/CREDIT/BOTH)
2. Build exact lookup table from training corpus
3. Create Qdrant semantic index (direction-aware)
4. Train XGBoost with signal features + embeddings
5. Save all artefacts for orchestrator

Usage:
    python scripts/train_models.py --version v1.0
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import yaml
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import re
from rapidfuzz import fuzz, process

# ────────────────────────────────────────────────
# CONFIG & UTILITIES
# ────────────────────────────────────────────────

def load_config():
    """Load main configuration file"""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("config/config.yaml not found")
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_feature_signal_config():
    """Load feature signals configuration"""
    config_path = Path("config/feature_signals.json")
    if not config_path.exists():
        raise FileNotFoundError("config/feature_signals.json not found")
    with open(config_path) as f:
        return json.load(f)

def load_category_tree(version: str):
    """Load category tree with direction rules"""
    tree_path = Path(f"data/versions/{version}/category_tree.json")
    if not tree_path.exists():
        raise FileNotFoundError(f"category_tree.json not found for version {version}")
    with open(tree_path) as f:
        return json.load(f)

def build_direction_rules_from_tree(category_tree: Dict) -> Dict:
    """Extract direction rules from category tree
    
    Returns:
        Dict mapping category -> direction (DEBIT/CREDIT/BOTH)
    """
    rules = {}
    
    def recurse(node):
        if 'category' in node and 'transaction_direction' in node:
            cat = node['category']
            dir_val = node['transaction_direction'].upper()
            if dir_val in ('CREDIT', 'DEBIT', 'BOTH'):
                rules[cat] = dir_val
        if 'subcategories' in node:
            for sub in node['subcategories']:
                recurse(sub)
    
    tree_root = category_tree.get('category_tree', {})
    for top_level in tree_root.values():
        recurse(top_level)
    
    print(f"✓ Built {len(rules)} direction rules from category tree")
    return rules

def build_category_keyword_map(signal_config: Dict, direction_rules: Dict) -> Dict:
    """Build mapping of category -> list of keywords for fast lookup
    
    Returns:
        Dict[category] -> List[keywords]
    """
    cat_keywords = {}
    
    for feat in signal_config.get('keyword_features', []):
        cat = feat.get('hint_category')
        if cat and cat in direction_rules:
            if cat not in cat_keywords:
                cat_keywords[cat] = set()
            cat_keywords[cat].update(feat['keywords'])
    
    for feat in signal_config.get('regex_features', []):
        cat = feat.get('hint_category')
        if cat and cat in direction_rules:
            if cat not in cat_keywords:
                cat_keywords[cat] = set()
            # Extract simple text patterns from regex where possible
            pattern = feat['pattern']
            # Simple extraction - you may want to enhance this
            simple_text = re.sub(r'[\\^$*+?.\[\]{}()|]', '', pattern).lower()
            if simple_text:
                cat_keywords[cat].add(simple_text)
    
    for feat in signal_config.get('composite_features', []):
        cat = feat.get('hint_category')
        if cat and cat in direction_rules:
            if cat not in cat_keywords:
                cat_keywords[cat] = set()
            for cond in feat['conditions']:
                if cond.get('match_type') == 'any_contains':
                    cat_keywords[cat].update(cond.get('keywords', []))
    
    # Convert sets to lists
    cat_keywords = {k: list(v) for k, v in cat_keywords.items()}
    
    print(f"✓ Built keyword map for {len(cat_keywords)} categories")
    return cat_keywords

# ────────────────────────────────────────────────
# DATA LOADING
# ────────────────────────────────────────────────

def load_training_data(version: str) -> pd.DataFrame:
    """Load training data CSV"""
    data_path = Path(f"data/versions/{version}/synthetic_data.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} training transactions")
    
    # Ensure required columns
    required = ['description', 'amount', 'transaction_type', 'category']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Normalise transaction_type
    df['transaction_type'] = df['transaction_type'].str.upper()
    
    return df

def normalise_description(text: str) -> str:
    """Normalise transaction description for matching"""
    text = str(text).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# ────────────────────────────────────────────────
# TIER 1: EXACT/FUZZY LOOKUP TABLE
# ────────────────────────────────────────────────

def build_exact_lookup_table(df: pd.DataFrame, direction_rules: Dict) -> Dict:
    """Build exact lookup table from training corpus
    
    Returns:
        Dict with structure:
        {
            'DEBIT': {normalised_description: category},
            'CREDIT': {normalised_description: category},
            'BOTH': {normalised_description: category}
        }
    """
    lookup = {'DEBIT': {}, 'CREDIT': {}, 'BOTH': {}}
    
    for _, row in df.iterrows():
        desc_norm = normalise_description(row['description'])
        category = row['category']
        tx_type = row['transaction_type']
        
        # Get category direction
        cat_direction = direction_rules.get(category, 'BOTH')
        
        # Add to appropriate lookup buckets
        if cat_direction == 'BOTH':
            lookup['BOTH'][desc_norm] = category
        elif cat_direction == tx_type:
            lookup[tx_type][desc_norm] = category
    
    total_entries = sum(len(v) for v in lookup.values())
    print(f"✓ Built exact lookup table with {total_entries} entries")
    print(f"  DEBIT: {len(lookup['DEBIT'])}, CREDIT: {len(lookup['CREDIT'])}, BOTH: {len(lookup['BOTH'])}")
    
    return lookup

# ────────────────────────────────────────────────
# TIER 2: EMBEDDING & QDRANT INDEX
# ────────────────────────────────────────────────

def init_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Initialise sentence transformer model"""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"✓ Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model

def generate_embeddings(df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    """Generate embeddings for all descriptions"""
    print("Generating embeddings...")
    descriptions = df['description'].apply(normalise_description).tolist()
    embeddings = model.encode(descriptions, show_progress_bar=True, batch_size=256)
    print(f"✓ Generated embeddings: {embeddings.shape}")
    return embeddings

def build_qdrant_index(df: pd.DataFrame, embeddings: np.ndarray, version: str, direction_rules: Dict):
    """Build Qdrant vector index with direction-aware payloads"""
    print("\nBuilding Qdrant index...")
    storage_path = Path(f"data/versions/{version}/qdrant_index")
    storage_path.mkdir(parents=True, exist_ok=True)
    
    client = QdrantClient(path=str(storage_path))
    collection_name = f"transactions_{version}"
    dim = embeddings.shape[1]
    
    # Delete existing collection
    try:
        client.delete_collection(collection_name)
        print(f"✓ Deleted existing collection: {collection_name}")
    except:
        pass
    
    # Create new collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"✓ Created collection: {collection_name}")
    
    # Prepare points
    points = []
    for idx, row in df.iterrows():
        category = row['category']
        tx_type = row['transaction_type']
        cat_direction = direction_rules.get(category, 'BOTH')
        
        points.append(PointStruct(
            id=idx,
            vector=embeddings[idx].tolist(),
            payload={
                "category": category,
                "primary_category": row.get('primary_category', category.split('_')[0]),
                "description": row['description'],
                "amount": float(row['amount']),
                "transaction_type": tx_type,
                "transaction_direction": cat_direction,
            }
        ))
    
    # Upload in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(collection_name, points=batch)
        if (i + batch_size) % 1000 == 0:
            print(f"  Uploaded {i + batch_size}/{len(points)} points")
    
    print(f"✓ Uploaded {len(points)} points to Qdrant")
    
    # Create payload indices for fast filtering
    for field in ["category", "transaction_direction"]:
        client.create_payload_index(collection_name, field_name=field, field_schema="keyword")
    
    print("✓ Created payload indices")
    return client, collection_name

# ────────────────────────────────────────────────
# TIER 3: FEATURE EXTRACTION FOR XGBOOST
# ────────────────────────────────────────────────

def extract_signal_features(row: pd.Series, signal_config: Dict, direction_rules: Dict) -> Dict:
    """Extract signal-based features with direction awareness"""
    signals = {}
    description = str(row['description']).lower()
    amount = float(row['amount'])
    tx_type = str(row['transaction_type']).upper()
    
    # Keyword features
    for feat in signal_config.get('keyword_features', []):
        name = feat['feature_name']
        cat = feat.get('hint_category', '')
        
        # Direction guard
        expected_dir = direction_rules.get(cat)
        if expected_dir and expected_dir != 'BOTH' and tx_type != expected_dir:
            signals[name] = 0.0
            continue
        
        # Check keywords
        matched = any(kw.lower() in description for kw in feat['keywords'])
        signals[name] = feat.get('confidence', 0.85) if matched else 0.0
    
    # Regex features
    for feat in signal_config.get('regex_features', []):
        name = feat['feature_name']
        cat = feat.get('hint_category', '')
        
        # Direction guard
        expected_dir = direction_rules.get(cat)
        if expected_dir and expected_dir != 'BOTH' and tx_type != expected_dir:
            signals[name] = 0.0
            continue
        
        # Check pattern
        matched = bool(re.search(feat['pattern'], description, re.IGNORECASE))
        signals[name] = feat.get('confidence', 0.85) if matched else 0.0
    
    # Composite features
    for feat in signal_config.get('composite_features', []):
        name = feat['feature_name']
        
        all_met = True
        for cond in feat['conditions']:
            if cond['match_type'] == 'any_contains':
                if not any(kw.lower() in description for kw in cond.get('keywords', [])):
                    all_met = False
                    break
        
        signals[name] = 1.0 if all_met else 0.0
    
    return signals

def build_feature_matrix(
    df: pd.DataFrame, 
    embeddings: np.ndarray,
    client: QdrantClient,
    collection_name: str,
    signal_config: Dict,
    direction_rules: Dict
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build feature matrix for XGBoost training"""
    print("\nBuilding feature matrix for XGBoost...")
    
    feature_list = []
    labels = []
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Processed {idx}/{len(df)} rows...")
        
        # Extract signal features
        signals = extract_signal_features(row, signal_config, direction_rules)
        
        # Get similar transactions from Qdrant (excluding self)
        # Note: We don't apply direction filter here during training as we want the model
        # to learn from all patterns. Direction filtering happens during inference.
        results = client.query_points(
            collection_name=collection_name,
            query=embeddings[idx].tolist(),
            limit=5
        ).points
        
        # Filter out self-matches
        similar = [r for r in results if r.id != idx][:3]
        
        # Build feature vector
        features = {}
        features.update(signals)
        features['total_signals_fired'] = sum(signals.values())
        
        # Similarity scores
        features['top1_score'] = similar[0].score if len(similar) > 0 else 0.0
        features['top2_score'] = similar[1].score if len(similar) > 1 else 0.0
        features['top3_score'] = similar[2].score if len(similar) > 2 else 0.0
        features['score_gap'] = features['top1_score'] - features['top2_score']
        
        # Amount features
        amount = float(row['amount'])
        tx_type = row['transaction_type']
        
        features['amount_log'] = np.log1p(abs(amount))
        features['amount_abs'] = abs(amount)
        features['is_debit'] = 1 if tx_type == 'DEBIT' else 0
        features['amount_log_credit'] = np.log1p(amount) if tx_type == 'CREDIT' else 0.0
        features['amount_log_debit'] = np.log1p(abs(amount)) if tx_type == 'DEBIT' else 0.0
        features['large_credit'] = 1 if tx_type == 'CREDIT' and amount > 5000 else 0
        features['large_debit'] = 1 if tx_type == 'DEBIT' and abs(amount) > 5000 else 0
        
        # Text features
        description = str(row['description'])
        features['word_count'] = len(description.split())
        features['char_count'] = len(description)
        
        # Embedding features (first 50 dimensions)
        for i in range(min(50, embeddings.shape[1])):
            features[f'emb_{i}'] = float(embeddings[idx][i])
        
        feature_list.append(features)
        labels.append(row['category'])
    
    # Convert to arrays
    df_features = pd.DataFrame(feature_list).fillna(0)
    X = df_features.values
    y = np.array(labels)
    feature_names = df_features.columns.tolist()
    
    print(f"\n✓ Feature matrix built:")
    print(f"  Shape: {X.shape}")
    print(f"  Categories: {len(np.unique(y))}")
    print(f"  Features: {len(feature_names)}")
    
    return X, y, feature_names

# ────────────────────────────────────────────────
# XGBOOST TRAINING
# ────────────────────────────────────────────────

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str]
):
    """Train XGBoost classifier"""
    print("\nTraining XGBoost classifier...")
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train_enc,
        eval_set=[(X_test, y_test_enc)],
        verbose=True
    )
    
    # Evaluate
    y_pred_enc = model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✓ Test Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Feature importance
    print("\nTop 20 Features by Importance:")
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    print(importance_df.to_string(index=False))
    
    # Store label encoder in model
    model._label_encoder = le
    
    return model, accuracy

# ────────────────────────────────────────────────
# SAVE ARTEFACTS
# ────────────────────────────────────────────────

def save_artefacts(
    version: str,
    model: xgb.XGBClassifier,
    feature_names: List[str],
    accuracy: float,
    signal_config: Dict,
    direction_rules: Dict,
    exact_lookup: Dict,
    category_keywords: Dict,
    collection_name: str
):
    """Save all training artefacts"""
    print("\nSaving training artefacts...")
    
    base_path = Path(f"data/versions/{version}")
    base_path.mkdir(parents=True, exist_ok=True)
    
    # XGBoost model
    model.save_model(str(base_path / "xgboost_model.json"))
    print("✓ Saved XGBoost model")
    
    # Label encoder
    with open(base_path / "label_encoder.json", 'w') as f:
        json.dump(
            {int(k): str(v) for k, v in enumerate(model._label_encoder.classes_)},
            f, indent=2
        )
    print("✓ Saved label encoder")
    
    # Feature names
    with open(base_path / "feature_names.json", 'w') as f:
        json.dump(feature_names, f, indent=2)
    print("✓ Saved feature names")
    
    # Direction rules
    with open(base_path / "direction_rules.json", 'w') as f:
        json.dump(direction_rules, f, indent=2)
    print("✓ Saved direction rules")
    
    # Exact lookup table
    with open(base_path / "exact_lookup.json", 'w') as f:
        json.dump(exact_lookup, f, indent=2)
    print("✓ Saved exact lookup table")
    
    # Category keywords map
    with open(base_path / "category_keywords.json", 'w') as f:
        json.dump(category_keywords, f, indent=2)
    print("✓ Saved category keywords map")
    
    # Metadata
    metadata = {
        "version": version,
        "trained_at": datetime.now().isoformat(),
        "accuracy": float(accuracy),
        "embedding_model": "all-MiniLM-L6-v2",
        "qdrant_collection": collection_name,
        "num_features": len(feature_names),
        "num_categories": len(model._label_encoder.classes_),
        "signal_stats": {
            "keyword_features": len(signal_config.get('keyword_features', [])),
            "regex_features": len(signal_config.get('regex_features', [])),
            "composite_features": len(signal_config.get('composite_features', []))
        },
        "direction_rules_count": len(direction_rules),
        "exact_lookup_entries": sum(len(v) for v in exact_lookup.values()),
        "categories_with_keywords": len(category_keywords)
    }
    
    with open(base_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✓ Saved metadata")
    
    print(f"\n✓ All artefacts saved to: {base_path}")

# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SME transaction categorisation models")
    parser.add_argument("--version", default="v1.0", help="Version identifier")
    args = parser.parse_args()
    
    print("=" * 70)
    print("SME TRANSACTION CATEGORISATION - MODEL TRAINING")
    print("=" * 70)
    print(f"Version: {args.version}\n")
    
    # Load configurations
    config = load_config()
    signal_config = load_feature_signal_config()
    category_tree = load_category_tree(args.version)
    
    # Build direction rules and keyword map
    direction_rules = build_direction_rules_from_tree(category_tree)
    category_keywords = build_category_keyword_map(signal_config, direction_rules)
    
    # Load training data
    df = load_training_data(args.version)
    
    # Build Tier 1: Exact lookup table
    exact_lookup = build_exact_lookup_table(df, direction_rules)
    
    # Build Tier 2: Semantic index
    embedding_model = init_embedding_model()
    embeddings = generate_embeddings(df, embedding_model)
    client, collection_name = build_qdrant_index(df, embeddings, args.version, direction_rules)
    
    # Build Tier 3: XGBoost training data
    X, y, feature_names = build_feature_matrix(
        df, embeddings, client, collection_name, signal_config, direction_rules
    )
    
    # Train/test split
    test_size = config.get('training', {}).get('test_split', 0.2)
    random_seed = config.get('training', {}).get('random_seed', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y
    )
    
    print(f"\nTrain/Test Split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Train XGBoost
    model, accuracy = train_xgboost(X_train, y_train, X_test, y_test, feature_names)
    
    # Save all artefacts
    save_artefacts(
        args.version,
        model,
        feature_names,
        accuracy,
        signal_config,
        direction_rules,
        exact_lookup,
        category_keywords,
        collection_name
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"Artefacts saved to: data/versions/{args.version}/")
    print("=" * 70)

if __name__ == "__main__":
    main()