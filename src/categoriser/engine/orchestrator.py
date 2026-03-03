#!/usr/bin/env python3
"""
Transaction Categorisation Orchestrator

Real-time transaction categorisation using tiered approach:
1. Tier 1: Exact/Fuzzy match (fast, high precision)
2. Tier 2: Keyword-filtered semantic search (direction-aware)
3. Tier 3: XGBoost with signal features (complex cases)

All tiers respect direction guardrails (DEBIT/CREDIT/BOTH)

Usage:
    python scripts/categorise_transactions.py --version v1.0 --input data/test.csv --output results.csv
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import xgboost as xgb
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

# Thresholds
FUZZY_MATCH_THRESHOLD = 0.90
KEYWORD_HIGH_CONFIDENCE_THRESHOLD = 0.92  # If keyword match is this strong, accept immediately
KEYWORD_MEDIUM_CONFIDENCE_THRESHOLD = 0.75  # If above this, use semantic as validation
SEMANTIC_CONFIDENCE_THRESHOLD = 0.65
SEMANTIC_GAP_THRESHOLD = 0.10

# Debug mode
DEBUG_MODE = True  # Set to False in production

def debug_print(message: str):
    """Print debug messages if debug mode is enabled"""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

class TransactionCategoriser:
    """Main categorisation engine with tiered approach"""
    
    def __init__(self, version: str):
        self.version = version
        self.version_path = Path(f"data/versions/{version}")
        
        print(f"Initialising categoriser for version: {version}")
        self._load_all_artefacts()
        
        self.stats = {
            'total': 0,
            'tier1_exact': 0,
            'tier1_fuzzy': 0,
            'tier2_keyword_direct': 0,  # High confidence keyword, no semantic needed
            'tier2_keyword_semantic': 0,  # Keyword + semantic validation
            'tier2_full_semantic': 0,
            'tier3_xgboost': 0,
            'unknown': 0,
            'tier1_time_ms': [],
            'tier2_time_ms': [],
            'tier3_time_ms': []
        }
        
        print("✓ Categoriser ready!\n")

    # ────────────────────────────────────────────────
    # LOADING ARTEFACTS
    # ────────────────────────────────────────────────

    def _load_all_artefacts(self):
        """Load all trained artefacts"""
        print("\nLoading artefacts...")
        
        # Config files
        with open("config/config.yaml") as f:
            self.config = yaml.safe_load(f)
        
        with open("config/feature_signals.json") as f:
            self.signal_config = json.load(f)
        print("  ✓ Loaded config files")
        
        # Direction rules
        with open(self.version_path / "direction_rules.json") as f:
            self.direction_rules = json.load(f)
        print(f"  ✓ Loaded {len(self.direction_rules)} direction rules")
        
        # Exact lookup table
        with open(self.version_path / "exact_lookup.json") as f:
            self.exact_lookup = json.load(f)
        total_exact = sum(len(v) for v in self.exact_lookup.values())
        print(f"  ✓ Loaded exact lookup table ({total_exact} entries)")
        
        # Category keywords
        with open(self.version_path / "category_keywords.json") as f:
            self.category_keywords = json.load(f)
        print(f"  ✓ Loaded keywords for {len(self.category_keywords)} categories")
        
        # Debug: Show sample keywords
        debug_print(f"Sample category keywords: {list(self.category_keywords.keys())[:5]}")
        
        # Embedding model
        print("  Loading embedding model...")
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("  ✓ Loaded embedding model")
        
        # Qdrant client
        qdrant_path = self.version_path / "qdrant_index"
        self.qdrant_client = QdrantClient(path=str(qdrant_path))
        
        with open(self.version_path / "metadata.json") as f:
            metadata = json.load(f)
            self.collection_name = metadata['qdrant_collection']
        print(f"  ✓ Loaded Qdrant collection: {self.collection_name}")
        
        # XGBoost model
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(str(self.version_path / "xgboost_model.json"))
        
        with open(self.version_path / "label_encoder.json") as f:
            label_mapping = json.load(f)
            self.label_encoder = {int(k): v for k, v in label_mapping.items()}
        
        with open(self.version_path / "feature_names.json") as f:
            self.feature_names = json.load(f)
        print(f"  ✓ Loaded XGBoost model ({len(self.feature_names)} features)")

    # ────────────────────────────────────────────────
    # UTILITIES
    # ────────────────────────────────────────────────

    def normalise_description(self, text: str) -> str:
        """Normalise description for matching"""
        text = str(text).strip().lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def get_allowed_categories(self, transaction_type: str) -> List[str]:
        """Get list of allowed categories based on transaction direction
        
        Args:
            transaction_type: DEBIT or CREDIT
            
        Returns:
            List of category names that match the direction
        """
        allowed = []
        tx_type = transaction_type.upper()
        
        for category, direction in self.direction_rules.items():
            if direction == 'BOTH' or direction == tx_type:
                allowed.append(category)
        
        debug_print(f"Allowed categories for {tx_type}: {len(allowed)} categories")
        return allowed

    # ────────────────────────────────────────────────
    # TIER 1: EXACT & FUZZY MATCHING
    # ────────────────────────────────────────────────

    def tier1_exact_fuzzy_match(
        self, 
        description: str, 
        transaction_type: str
    ) -> Optional[Tuple[str, float, str]]:
        """
        Tier 1: Exact and fuzzy matching against lookup table
        
        Returns:
            (category, confidence, method) or None
        """
        desc_norm = self.normalise_description(description)
        tx_type = transaction_type.upper()
        
        debug_print(f"Tier 1: Normalised description: '{desc_norm}'")
        
        # Build search space (direction-filtered)
        search_space = {}
        
        # Add BOTH categories
        search_space.update(self.exact_lookup.get('BOTH', {}))
        
        # Add direction-specific categories
        if tx_type in ('DEBIT', 'CREDIT'):
            search_space.update(self.exact_lookup.get(tx_type, {}))
        
        debug_print(f"Tier 1: Search space size: {len(search_space)} entries")
        
        if not search_space:
            debug_print("Tier 1: No search space available")
            return None
        
        # Try exact match first
        if desc_norm in search_space:
            category = search_space[desc_norm]
            debug_print(f"Tier 1: EXACT MATCH found -> {category}")
            return (category, 1.0, 'tier1_exact')
        
        # Try fuzzy match
        match = process.extractOne(
            desc_norm,
            search_space.keys(),
            scorer=fuzz.ratio,
            score_cutoff=FUZZY_MATCH_THRESHOLD * 100
        )
        
        if match:
            matched_desc, score, _ = match
            category = search_space[matched_desc]
            confidence = score / 100.0
            debug_print(f"Tier 1: FUZZY MATCH found -> {category} (score: {confidence:.4f}, matched: '{matched_desc}')")
            return (category, confidence, 'tier1_fuzzy')
        
        debug_print("Tier 1: No match found")
        return None

    # ────────────────────────────────────────────────
    # TIER 2: KEYWORD-FILTERED SEMANTIC SEARCH
    # ────────────────────────────────────────────────

    def find_keyword_matched_categories(
        self, 
        description: str, 
        allowed_categories: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Find categories whose keywords appear in the description (with fuzzy matching for typos)
        
        Args:
            description: Transaction description (normalised)
            allowed_categories: Direction-filtered category list
            
        Returns:
            List of tuples (category, match_score) sorted by score descending
        """
        desc_lower = description.lower()
        matches = []  # List of (category, score)
        
        debug_print(f"Tier 2 Keyword: Searching in '{desc_lower}'")
        
        for category in allowed_categories:
            keywords = self.category_keywords.get(category, [])
            if not keywords:
                continue
            
            best_score = 0.0
            best_keyword = None
            
            for kw in keywords:
                kw_lower = kw.lower()
                
                # Try exact substring match first (fastest)
                if kw_lower in desc_lower:
                    score = 100.0
                    if score > best_score:
                        best_score = score
                        best_keyword = kw
                    continue
                
                # Try fuzzy match for typos (e.g., "COURUIER" vs "courier")
                fuzzy_score = fuzz.partial_ratio(kw_lower, desc_lower)
                if fuzzy_score > 85 and fuzzy_score > best_score:
                    best_score = fuzzy_score
                    best_keyword = kw
            
            if best_score > 0:
                match_confidence = best_score / 100.0
                matches.append((category, match_confidence))
                debug_print(f"Tier 2 Keyword: Match '{best_keyword}' -> {category} (score: {best_score:.2f})")
        
        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        
        debug_print(f"Tier 2 Keyword: Found {len(matches)} category matches")
        return matches

    def tier2_semantic_search(
        self,
        description: str,
        transaction_type: str,
        amount: float
    ) -> Optional[Tuple[str, float, str, List[Dict]]]:
        """
        Tier 2: Keyword-first approach with semantic validation
        
        Logic:
        1. Find keyword matches with scores
        2. If ANY keyword match > 92% → Accept immediately (no semantic needed)
        3. If keyword matches 75-92% → Use semantic search within those categories as validation
        4. If no strong keywords → Full semantic search across all allowed categories
        
        Returns:
            (category, confidence, method, top3_results) or None
        """
        # Get direction-filtered categories
        allowed_categories = self.get_allowed_categories(transaction_type)
        
        if not allowed_categories:
            debug_print("Tier 2: No allowed categories")
            return None
        
        # Find keyword matches with scores
        keyword_matches = self.find_keyword_matched_categories(
            self.normalise_description(description), 
            allowed_categories
        )
        
        # Check for high-confidence keyword match (>92%)
        if keyword_matches and keyword_matches[0][1] >= KEYWORD_HIGH_CONFIDENCE_THRESHOLD:
            category, confidence = keyword_matches[0]
            debug_print(f"Tier 2: HIGH CONFIDENCE keyword match -> {category} (score: {confidence:.4f})")
            debug_print(f"Tier 2: Accepting immediately without semantic search")
            return (category, confidence, 'tier2_keyword_direct', [])
        
        # Generate embedding for semantic search
        desc_norm = self.normalise_description(description)
        embedding = self.embedding_model.encode([desc_norm])[0]
        
        # Determine search strategy
        if keyword_matches and keyword_matches[0][1] >= KEYWORD_MEDIUM_CONFIDENCE_THRESHOLD:
            # Medium confidence keywords (75-92%) - use semantic as validation
            keyword_matched_cats = [cat for cat, score in keyword_matches]
            category_filter = Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchAny(any=keyword_matched_cats)
                    )
                ]
            )
            method_prefix = 'tier2_keyword_semantic'
            debug_print(f"Tier 2: Medium confidence keywords - using semantic validation")
            debug_print(f"Tier 2: Searching within {len(keyword_matched_cats)} keyword-matched categories")
        else:
            # No strong keywords - full semantic search
            category_filter = Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchAny(any=allowed_categories)
                    )
                ]
            )
            method_prefix = 'tier2_full_semantic'
            debug_print(f"Tier 2: No strong keywords - full semantic search ({len(allowed_categories)} categories)")
        
        # Query Qdrant
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=embedding.tolist(),
            limit=10,
            query_filter=category_filter
        ).points
        
        if not results:
            debug_print("Tier 2: No Qdrant results")
            return None
        
        # Extract top results
        top3 = [
            {
                'category': r.payload['category'],
                'score': r.score,
                'description': r.payload.get('description', '')
            }
            for r in results[:3]
        ]
        
        top1_score = results[0].score
        top2_score = results[1].score if len(results) > 1 else 0.0
        score_gap = top1_score - top2_score
        
        debug_print(f"Tier 2: Top semantic result: {results[0].payload['category']} (score: {top1_score:.4f})")
        debug_print(f"Tier 2: Score gap: {score_gap:.4f}")
        
        # Decision logic
        if top1_score >= SEMANTIC_CONFIDENCE_THRESHOLD and score_gap >= SEMANTIC_GAP_THRESHOLD:
            category = results[0].payload['category']
            confidence = top1_score
            debug_print(f"Tier 2: ACCEPTED -> {category} (confidence: {confidence:.4f})")
            return (category, confidence, method_prefix, top3)
        
        debug_print(f"Tier 2: REJECTED (score: {top1_score:.4f} < {SEMANTIC_CONFIDENCE_THRESHOLD} or gap: {score_gap:.4f} < {SEMANTIC_GAP_THRESHOLD})")
        return None

    # ────────────────────────────────────────────────
    # TIER 3: XGBOOST WITH DIRECTION GUARD
    # ────────────────────────────────────────────────

    def extract_signal_features(self, transaction: Dict) -> Dict:
        """Extract signal features from transaction"""
        signals = {}
        description = str(transaction['description']).lower()
        amount = float(transaction['amount'])
        tx_type = str(transaction.get('transaction_type', 'UNKNOWN')).upper()
        
        # Keyword features
        for feat in self.signal_config.get('keyword_features', []):
            name = feat['feature_name']
            cat = feat.get('hint_category', '')
            
            # Direction guard
            expected_dir = self.direction_rules.get(cat)
            if expected_dir and expected_dir != 'BOTH' and tx_type != expected_dir:
                signals[name] = 0.0
                continue
            
            matched = any(kw.lower() in description for kw in feat['keywords'])
            signals[name] = feat.get('confidence', 0.85) if matched else 0.0
        
        # Regex features
        for feat in self.signal_config.get('regex_features', []):
            name = feat['feature_name']
            cat = feat.get('hint_category', '')
            
            expected_dir = self.direction_rules.get(cat)
            if expected_dir and expected_dir != 'BOTH' and tx_type != expected_dir:
                signals[name] = 0.0
                continue
            
            matched = bool(re.search(feat['pattern'], description, re.IGNORECASE))
            signals[name] = feat.get('confidence', 0.85) if matched else 0.0
        
        # Composite features
        for feat in self.signal_config.get('composite_features', []):
            name = feat['feature_name']
            all_met = all(
                any(kw.lower() in description for kw in cond.get('keywords', []))
                for cond in feat['conditions'] if cond['match_type'] == 'any_contains'
            )
            signals[name] = 1.0 if all_met else 0.0
        
        return signals

    def build_feature_vector(
        self,
        transaction: Dict,
        embedding: np.ndarray,
        qdrant_top3: List[Dict],
        signals: Dict
    ) -> np.ndarray:
        """Build feature vector for XGBoost prediction"""
        features = {}
        
        # Signal features
        features.update(signals)
        features['total_signals_fired'] = sum(signals.values())
        
        # Semantic similarity features
        features['top1_score'] = qdrant_top3[0]['score'] if len(qdrant_top3) > 0 else 0.0
        features['top2_score'] = qdrant_top3[1]['score'] if len(qdrant_top3) > 1 else 0.0
        features['top3_score'] = qdrant_top3[2]['score'] if len(qdrant_top3) > 2 else 0.0
        features['score_gap'] = features['top1_score'] - features['top2_score']
        
        # Amount features
        amount = float(transaction['amount'])
        tx_type = str(transaction.get('transaction_type', '')).upper()
        
        features['amount_log'] = np.log1p(abs(amount))
        features['amount_abs'] = abs(amount)
        features['is_debit'] = 1 if tx_type == 'DEBIT' else 0
        features['amount_log_credit'] = np.log1p(amount) if tx_type == 'CREDIT' else 0.0
        features['amount_log_debit'] = np.log1p(abs(amount)) if tx_type == 'DEBIT' else 0.0
        features['large_credit'] = 1 if tx_type == 'CREDIT' and amount > 5000 else 0
        features['large_debit'] = 1 if tx_type == 'DEBIT' and abs(amount) > 5000 else 0
        
        # Text features
        description = str(transaction['description'])
        features['word_count'] = len(description.split())
        features['char_count'] = len(description)
        
        # Embedding features
        for i in range(min(50, len(embedding))):
            features[f'emb_{i}'] = float(embedding[i])
        
        # Build vector in correct order
        feature_vector = [features.get(fname, 0.0) for fname in self.feature_names]
        return np.array(feature_vector).reshape(1, -1)

    def tier3_xgboost_predict(
        self,
        transaction: Dict,
        embedding: np.ndarray,
        qdrant_top3: List[Dict]
    ) -> Tuple[str, float, str]:
        """
        Tier 3: XGBoost prediction with direction guard
        
        Returns:
            (category, confidence, method)
        """
        debug_print("Tier 3: Falling back to XGBoost")
        
        # Extract signals
        signals = self.extract_signal_features(transaction)
        
        # Build feature vector
        feature_vector = self.build_feature_vector(transaction, embedding, qdrant_top3, signals)
        
        # Get prediction probabilities
        probs = self.xgb_model.predict_proba(feature_vector)[0]
        
        # Apply direction guard
        tx_type = str(transaction.get('transaction_type', '')).upper()
        allowed_categories = self.get_allowed_categories(tx_type)
        
        # Mask out disallowed categories
        all_categories = list(self.label_encoder.values())
        masked_probs = probs.copy()
        
        for i, cat in enumerate(all_categories):
            if cat not in allowed_categories:
                masked_probs[i] = 0.0
        
        # Normalise
        if masked_probs.sum() > 0:
            masked_probs /= masked_probs.sum()
        
        # Get best prediction
        best_idx = np.argmax(masked_probs)
        category = all_categories[best_idx]
        confidence = float(masked_probs[best_idx])
        
        debug_print(f"Tier 3: Predicted {category} with confidence {confidence:.4f}")
        
        return (category, confidence, 'tier3_xgboost')

    # ────────────────────────────────────────────────
    # MAIN CATEGORISATION
    # ────────────────────────────────────────────────

    def categorise_transaction(self, transaction: Dict) -> Dict:
        """
        Categorise a single transaction using tiered approach
        
        Args:
            transaction: Dict with keys: description, amount, transaction_type
            
        Returns:
            Dict with categorisation result
        """
        start_time = datetime.now()
        
        description = transaction['description']
        amount = float(transaction['amount'])
        tx_type = str(transaction.get('transaction_type', 'UNKNOWN')).upper()
        
        debug_print(f"\n{'='*70}")
        debug_print(f"Categorising: '{description}' | {tx_type} | £{amount}")
        debug_print(f"{'='*70}")
        
        # Validate transaction type
        if tx_type not in ('DEBIT', 'CREDIT'):
            debug_print(f"Invalid transaction type: {tx_type}")
            return {
                'category': 'UNKNOWN',
                'confidence': 0.0,
                'method': 'invalid_transaction_type',
                'processing_time_ms': 0.0,
                'tier': 'none'
            }
        
        result = None
        qdrant_top3 = []
        
        # TIER 1: Exact/Fuzzy matching
        tier1_start = datetime.now()
        result = self.tier1_exact_fuzzy_match(description, tx_type)
        tier1_time = (datetime.now() - tier1_start).total_seconds() * 1000
        
        if result:
            category, confidence, method = result
            tier = 'tier1'
            self.stats['tier1_time_ms'].append(tier1_time)
            if 'exact' in method:
                self.stats['tier1_exact'] += 1
            else:
                self.stats['tier1_fuzzy'] += 1
        else:
            # TIER 2: Keyword-first with semantic validation
            tier2_start = datetime.now()
            result = self.tier2_semantic_search(description, tx_type, amount)
            tier2_time = (datetime.now() - tier2_start).total_seconds() * 1000
            
            if result:
                category, confidence, method, qdrant_top3 = result
                tier = 'tier2'
                self.stats['tier2_time_ms'].append(tier2_time)
                if 'keyword_direct' in method:
                    self.stats['tier2_keyword_direct'] += 1
                elif 'keyword' in method:
                    self.stats['tier2_keyword_semantic'] += 1
                else:
                    self.stats['tier2_full_semantic'] += 1
            else:
                # TIER 3: XGBoost
                tier3_start = datetime.now()
                
                # Get embedding and Qdrant results for features
                desc_norm = self.normalise_description(description)
                embedding = self.embedding_model.encode([desc_norm])[0]
                
                # Get Qdrant results for feature extraction
                allowed_cats = self.get_allowed_categories(tx_type)
                if allowed_cats:
                    qdrant_results = self.qdrant_client.query_points(
                        collection_name=self.collection_name,
                        query=embedding.tolist(),
                        limit=3,
                        query_filter=Filter(
                            must=[FieldCondition(key="category", match=MatchAny(any=allowed_cats))]
                        )
                    ).points
                    
                    qdrant_top3 = [
                        {'category': r.payload['category'], 'score': r.score}
                        for r in qdrant_results
                    ]
                    
                    # Debug output with cleaner formatting
                    top3_str = ", ".join([f"{r['category']} ({r['score']:.4f})" for r in qdrant_top3])
                    debug_print(f"Tier 3: Qdrant top 3 for features: {top3_str}")
                
                category, confidence, method = self.tier3_xgboost_predict(
                    transaction, embedding, qdrant_top3
                )
                
                tier = 'tier3'
                tier3_time = (datetime.now() - tier3_start).total_seconds() * 1000
                self.stats['tier3_time_ms'].append(tier3_time)
                self.stats['tier3_xgboost'] += 1
        
        # Calculate total time
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Update stats
        self.stats['total'] += 1
        
        debug_print(f"FINAL RESULT: {category} | Confidence: {confidence:.4f} | Tier: {tier}")
        debug_print(f"{'='*70}\n")
        
        return {
            'category': category,
            'confidence': round(confidence, 4),
            'method': method,
            'tier': tier,
            'processing_time_ms': round(total_time, 2),
            'qdrant_top3': qdrant_top3
        }

    def categorise_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Categorise a batch of transactions"""
        results = []
        total = len(transactions)
        
        print(f"\nCategorising {total} transactions...\n")
        
        for i, tx in enumerate(transactions):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{total}")
            
            result = self.categorise_transaction(tx)
            result['transaction_id'] = tx.get('id', i)
            result['description'] = tx['description']
            result['amount'] = tx['amount']
            result['transaction_type'] = tx.get('transaction_type', 'UNKNOWN')
            
            results.append(result)
        
        print(f"  ✓ Completed: {total}/{total}\n")
        return results

    def print_statistics(self):
        """Print categorisation statistics"""
        total = self.stats['total']
        if total == 0:
            print("No transactions categorised")
            return
        
        print("\n" + "=" * 70)
        print("CATEGORISATION STATISTICS")
        print("=" * 70)
        
        print(f"\nTotal Transactions: {total}")
        print(f"\nTier Breakdown:")
        print(f"  Tier 1 (Exact):           {self.stats['tier1_exact']:>6} ({self.stats['tier1_exact']/total*100:>5.1f}%)")
        print(f"  Tier 1 (Fuzzy):           {self.stats['tier1_fuzzy']:>6} ({self.stats['tier1_fuzzy']/total*100:>5.1f}%)")
        print(f"  Tier 2 (Keyword Direct):  {self.stats['tier2_keyword_direct']:>6} ({self.stats['tier2_keyword_direct']/total*100:>5.1f}%)")
        print(f"  Tier 2 (Keyword+Semantic):{self.stats['tier2_keyword_semantic']:>6} ({self.stats['tier2_keyword_semantic']/total*100:>5.1f}%)")
        print(f"  Tier 2 (Full Semantic):   {self.stats['tier2_full_semantic']:>6} ({self.stats['tier2_full_semantic']/total*100:>5.1f}%)")
        print(f"  Tier 3 (XGBoost):         {self.stats['tier3_xgboost']:>6} ({self.stats['tier3_xgboost']/total*100:>5.1f}%)")
        
        print(f"\nPerformance:")
        if self.stats['tier1_time_ms']:
            print(f"  Tier 1 avg: {np.mean(self.stats['tier1_time_ms']):>6.2f} ms")
        if self.stats['tier2_time_ms']:
            print(f"  Tier 2 avg: {np.mean(self.stats['tier2_time_ms']):>6.2f} ms")
        if self.stats['tier3_time_ms']:
            print(f"  Tier 3 avg: {np.mean(self.stats['tier3_time_ms']):>6.2f} ms")
        
        print("=" * 70)

# ────────────────────────────────────────────────
# CLI INTERFACE
# ────────────────────────────────────────────────

def load_test_data(input_path: Path) -> List[Dict]:
    """Load test transactions from CSV"""
    df = pd.read_csv(input_path)
    
    # Ensure required columns
    required = ['description', 'amount', 'transaction_type']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    transactions = df.to_dict('records')
    print(f"✓ Loaded {len(transactions)} test transactions")
    return transactions

def save_results(results: List[Dict], output_path: Path):
    """Save categorisation results to CSV"""
    df = pd.DataFrame(results)
    
    # Reorder columns for readability
    column_order = [
        'transaction_id', 'description', 'amount', 'transaction_type',
        'category', 'confidence', 'method', 'tier', 'processing_time_ms'
    ]
    
    existing_cols = [col for col in column_order if col in df.columns]
    other_cols = [col for col in df.columns if col not in column_order]
    
    df = df[existing_cols + other_cols]
    df.to_csv(output_path, index=False)
    
    print(f"✓ Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Categorise SME bank transactions using tiered approach"
    )
    parser.add_argument("--version", default="v1.0", help="Model version to use")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", default=None, help="Output CSV file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Set debug mode
    global DEBUG_MODE
    DEBUG_MODE = args.debug
    
    print("=" * 70)
    print("SME TRANSACTION CATEGORISATION - ORCHESTRATOR")
    print("=" * 70)
    print(f"Version: {args.version}")
    print(f"Input: {args.input}")
    print(f"Debug: {DEBUG_MODE}\n")
    
    # Initialise categoriser
    categoriser = TransactionCategoriser(args.version)
    
    # Load transactions
    transactions = load_test_data(Path(args.input))
    
    # Categorise
    results = categoriser.categorise_batch(transactions)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(f"data/categorisation_results_{timestamp}.csv")
    
    # Save results
    save_results(results, output_path)
    
    # Print statistics
    categoriser.print_statistics()
    
    # Show sample results
    print("\nSample Results (first 5):")
    print("-" * 70)
    for r in results[:5]:
        print(f"Description: {r['description'][:50]}")
        print(f"Category: {r['category']}")
        print(f"Confidence: {r['confidence']:.4f}")
        print(f"Method: {r['method']} ({r['tier']})")
        print(f"Time: {r['processing_time_ms']:.2f} ms")
        print("-" * 70)
    
    print("\n✓ Categorisation complete!")

if __name__ == "__main__":
    main()