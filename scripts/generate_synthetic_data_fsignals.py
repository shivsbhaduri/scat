#!/usr/bin/env python3
"""
Synthetic Transaction Data Generator - Enhanced

Forces descriptions to include strong keywords/patterns from feature_signals.json
for the target category → better Qdrant grounding and fast-path accuracy.

Usage:
    python scripts/generate_synthetic_data.py --version v1.0 --samples-per-category 100
"""

import json
import random
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import csv
import time
import yaml
from openai import OpenAI

# Load config
def load_config():
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("config/config.yaml not found")
    with open(config_path) as f:
        return yaml.safe_load(f)

# Initialise OpenRouter client
def init_openrouter_client(api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )

# Load category tree
def load_category_tree(version: str) -> Dict:
    tree_path = Path(f"data/versions/{version}/category_tree.json")
    if not tree_path.exists():
        raise FileNotFoundError(f"Category tree not found: {tree_path}")
    with open(tree_path) as f:
        return json.load(f)

# Load feature signals
def load_feature_signals():
    config_path = Path("config/feature_signals.json")
    if not config_path.exists():
        raise FileNotFoundError("feature_signals.json not found")
    with open(config_path) as f:
        return json.load(f)

# Extract leaf categories with full path
def extract_leaf_categories(category_tree: Dict) -> List[Dict]:
    leaf_categories = []
    
    for primary_key, primary_data in category_tree["category_tree"].items():
        primary_name = primary_data["name"]
        primary_direction = primary_data.get("transaction_direction", "BOTH")
        
        if "subcategories" in primary_data:
            for subcategory in primary_data["subcategories"]:
                leaf_categories.append({
                    "category": subcategory["category"],
                    "name": subcategory["name"],
                    "definition": subcategory["definition"],
                    "transaction_direction": subcategory.get("transaction_direction", primary_direction),
                    "synthetic_prompt": subcategory.get("synthetic_prompt", ""),
                    "primary_category": primary_key,
                    "primary_name": primary_name,
                    "full_category_path": f"{primary_name} > {subcategory['name']}"
                })
    
    return leaf_categories

# Build keyword/regex hints per category from feature_signals.json
def build_category_hints(feature_signals: Dict) -> Dict:
    hints = {}
    
    # Collect all keywords/regex for each hint_category
    for feat_type in ['keyword_features', 'regex_features']:
        for feat in feature_signals.get(feat_type, []):
            cat = feat.get('hint_category')
            if not cat:
                continue
            
            if cat not in hints:
                hints[cat] = {'keywords': [], 'patterns': []}
            
            if feat_type == 'keyword_features':
                hints[cat]['keywords'].extend(feat['keywords'])
            else:
                hints[cat]['patterns'].append(feat['pattern'])
    
    # Composite features also contribute keywords (from any_contains conditions)
    for feat in feature_signals.get('composite_features', []):
        cat = feat.get('hint_category')
        if not cat:
            continue
        
        if cat not in hints:
            hints[cat] = {'keywords': [], 'patterns': []}
        
        for cond in feat['conditions']:
            if cond['match_type'] == 'any_contains':
                hints[cat]['keywords'].extend(cond['keywords'])
    
    # Dedup and limit to top 5–10 strongest per category
    for cat in hints:
        hints[cat]['keywords'] = list(set(hints[cat]['keywords']))[:10]
        hints[cat]['patterns'] = list(set(hints[cat]['patterns']))[:5]
    
    return hints

# Load or initialise progress tracker
def load_progress(version: str) -> Dict:
    progress_path = Path(f"data/versions/{version}/generation_progress.json")
    if progress_path.exists():
        with open(progress_path) as f:
            return json.load(f)
    return {}

def save_progress(version: str, progress: Dict):
    progress_path = Path(f"data/versions/{version}/generation_progress.json")
    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)

# Generate account IDs pool
def generate_account_ids(count: int = 20) -> List[str]:
    return [f"ACC{random.randint(100000, 999999)}" for _ in range(count)]

# Check if CSV has header
def csv_has_header(file_path: Path) -> bool:
    if not file_path.exists():
        return False
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            return 'account_id' in first_line and 'transaction_id' in first_line
    except:
        return False

# Generate transactions using LLM with forced keywords/patterns
def generate_transactions_llm(
    client: OpenAI,
    model: str,
    category_data: Dict,
    category_hints: Dict,
    batch_size: int = 10,
    account_ids: List[str] = None
) -> List[Dict]:
    """Generate batch of transactions, forcing inclusion of category-specific keywords/patterns."""
    
    if account_ids is None:
        account_ids = generate_account_ids(20)
    
    cat = category_data['category']
    hints = category_hints.get(cat, {'keywords': [], 'patterns': []})
    direction = category_data['transaction_direction']
    
    forced_patterns = []
    if hints['keywords']:
        selected_keywords = random.sample(hints['keywords'], min(3, len(hints['keywords'])))
        forced_patterns.append(f"Include at least one of these keywords/phrases exactly or very closely: {', '.join(selected_keywords)}")
    if hints['patterns']:
        selected_patterns = random.sample(hints['patterns'], min(2, len(hints['patterns'])))
        forced_patterns.append(f"Match the style of at least one of these patterns: {', '.join(selected_patterns)}")
    
    forced_text = "\n".join(forced_patterns) if forced_patterns else "Use realistic description matching the synthetic_prompt."

    # Handle BOTH direction - tell LLM to generate mix of DEBIT and CREDIT
    if direction == "BOTH":
        direction_instruction = """transaction_type MUST be either "DEBIT" or "CREDIT" (randomly distribute across the batch)
- For DEBIT transactions: use descriptions that represent money going OUT (payments, purchases, fees, etc.)
- For CREDIT transactions: use descriptions that represent money coming IN (income, refunds, deposits, etc.)
- Generate a realistic mix of both directions for this category"""
        
        # We'll validate and fix after generation
        expected_direction = "DEBIT or CREDIT"
    else:
        direction_instruction = f'transaction_type MUST be "{direction}" for ALL transactions'
        expected_direction = direction

    prompt = f"""Generate {batch_size} realistic UK SME bank transactions for this category.

Category: {category_data['name']}
Category Code: {category_data['category']}
Definition: {category_data['definition']}
Transaction Direction: {category_data['transaction_direction']}

Guidance for generating realistic descriptions:
{category_data['synthetic_prompt']}

{forced_text}

CRITICAL RULES:
1. {direction_instruction}
2. amount must be a POSITIVE number (sign determined by transaction_type)
3. description must include at least one strong keyword/pattern from above (if provided)
4. description must make sense for the transaction_type (DEBIT = money out, CREDIT = money in)
5. date in YYYY-MM-DD within last 12 months
6. currency_code "GBP"

Return ONLY a valid JSON array with {batch_size} transactions, no other text, no explanations.

Format:
[
  {{
    "description": "realistic UK bank description",
    "amount": positive_number,
    "transaction_type": "DEBIT" or "CREDIT",
    "date": "YYYY-MM-DD",
    "currency_code": "GBP"
  }},
  ...
]
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a UK banking transaction data generator. Return valid JSON arrays only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
            max_tokens=3000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean markdown/code blocks
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].strip()
        content = content.strip('` \n')
        
        print(f"  Raw LLM response (first 200 chars): {content[:200]}...")  # DEBUG
        
        transactions = json.loads(content)
        
        # Add metadata + validate/fix direction
        for txn in transactions:
            txn['account_id'] = random.choice(account_ids)
            txn['transaction_id'] = f"TXN{random.randint(1000000, 9999999)}"
            txn['category'] = category_data['category']
            txn['primary_category'] = category_data['primary_category']
            txn['full_category_path'] = category_data['full_category_path']
            txn['confidence'] = 1.0
            
            # Fix transaction_type if needed
            if direction == "BOTH":
                # LLM should have generated DEBIT or CREDIT
                # If it generated "BOTH" (shouldn't happen), fix it
                if txn.get('transaction_type') == "BOTH":
                    txn['transaction_type'] = random.choice(["DEBIT", "CREDIT"])
                    print(f"WARNING: LLM generated 'BOTH' for transaction. Randomly assigned {txn['transaction_type']}")
                
                # Ensure it's valid
                if txn.get('transaction_type') not in ["DEBIT", "CREDIT"]:
                    txn['transaction_type'] = random.choice(["DEBIT", "CREDIT"])
                    print(f"WARNING: Invalid transaction_type. Randomly assigned {txn['transaction_type']}")
            
            else:
                # Direction is DEBIT or CREDIT - enforce it
                if txn.get('transaction_type') != direction:
                    print(f"WARNING: LLM generated wrong direction for {cat}. Expected {direction}, got {txn.get('transaction_type')}. Correcting.")
                    txn['transaction_type'] = direction
        
        print(f"  Generated {len(transactions)} valid transactions")  # DEBUG
        
        return transactions
    
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        print(f"  Bad content: {content[:500]}...")
        return []
    except Exception as e:
        print(f"ERROR generating for {cat}: {e}")
        return []

# Main generation function
def generate_synthetic_data(
    version: str,
    samples_per_category: int,
    batch_size: int = 10,
    config: Dict = None
):
    print(f"\nStarting synthetic data generation for version: {version}")
    print("=" * 60)
    
    # Load category tree & feature signals
    category_tree = load_category_tree(version)
    feature_signals = load_feature_signals()
    leaf_categories = extract_leaf_categories(category_tree)
    
    # Build hints per category from feature_signals
    category_hints = build_category_hints(feature_signals)
    print(f"Built hints for {len(category_hints)} categories from feature_signals.json")
    
    progress = load_progress(version)
    
    client = init_openrouter_client(config['openrouter']['api_key'])
    model = config['openrouter']['model']
    print(f"Using model: {model}")
    
    account_ids = generate_account_ids(20)
    
    output_path = Path(f"data/versions/{version}/synthetic_data.csv")
    has_header = csv_has_header(output_path)
    
    fieldnames = [
        'account_id', 'transaction_id', 'description', 'amount',
        'transaction_type', 'date', 'currency_code', 'category',
        'primary_category', 'full_category_path', 'confidence'
    ]
    
    csv_file = open(output_path, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    if not has_header:
        writer.writeheader()
    
    total_generated = 0
    total_categories = len(leaf_categories)
    
    try:
        for idx, category_data in enumerate(leaf_categories, 1):
            cat_code = category_data['category']
            cat_name = category_data['name']
            
            generated_so_far = progress.get(cat_code, {}).get('generated', 0)
            if generated_so_far >= samples_per_category:
                print(f"[{idx}/{total_categories}] Skipping {cat_name} (already complete)")
                continue
            
            remaining = samples_per_category - generated_so_far
            
            print(f"\n[{idx}/{total_categories}] {cat_name} ({cat_code})")
            print(f"  Direction: {category_data['transaction_direction']}")
            print(f"  Progress: {generated_so_far}/{samples_per_category}")
            print(f"  Remaining: {remaining}")
            
            category_generated = 0
            
            while category_generated < remaining:
                curr_batch = min(batch_size, remaining - category_generated)
                
                transactions = generate_transactions_llm(
                    client=client,
                    model=model,
                    category_data=category_data,
                    category_hints=category_hints,
                    batch_size=curr_batch,
                    account_ids=account_ids
                )
                
                if not transactions:
                    time.sleep(2)
                    continue
                
                for txn in transactions:
                    writer.writerow(txn)
                
                csv_file.flush()  # Force write to disk after each batch
                
                category_generated += len(transactions)
                total_generated += len(transactions)
                
                progress[cat_code] = {
                    'category': cat_name,
                    'generated': generated_so_far + category_generated,
                    'target': samples_per_category,
                    'completed': (generated_so_far + category_generated) >= samples_per_category,
                    'last_updated': datetime.now().isoformat()
                }
                save_progress(version, progress)
                
                print(f"  Generated & wrote {len(transactions)} (total: {generated_so_far + category_generated}/{samples_per_category})")
                
                time.sleep(0.5)
            
            print(f"  Completed {cat_name}")
    
    finally:
        csv_file.close()
    
    print("\nGENERATION COMPLETE")
    print(f"Total this run: {total_generated}")
    print(f"Output: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic transaction data")
    parser.add_argument("--version", default="v1.0")
    parser.add_argument("--samples-per-category", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    
    args = parser.parse_args()
    
    config = load_config()
    
    if not config.get('openrouter', {}).get('api_key'):
        raise ValueError("OpenRouter API key missing in config.yaml")
    
    generate_synthetic_data(
        version=args.version,
        samples_per_category=args.samples_per_category,
        batch_size=args.batch_size,
        config=config
    )

if __name__ == "__main__":
    main()