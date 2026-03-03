#!/usr/bin/env python3
"""
Synthetic Transaction Data Generator

Generates synthetic UK SME bank transactions using LLM based on category tree.
Uses OpenRouter API for LLM calls.
Supports incremental generation with progress tracking.

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

# Extract leaf categories with full path
def extract_leaf_categories(category_tree: Dict) -> List[Dict]:
    """Extract all leaf categories with their full path and metadata."""
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

# Load or initialise progress tracker
def load_progress(version: str) -> Dict:
    progress_path = Path(f"data/versions/{version}/generation_progress.json")
    
    if progress_path.exists():
        with open(progress_path) as f:
            return json.load(f)
    
    return {}

# Save progress
def save_progress(version: str, progress: Dict):
    progress_path = Path(f"data/versions/{version}/generation_progress.json")
    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)

# Generate account IDs pool
def generate_account_ids(count: int = 20) -> List[str]:
    """Generate a pool of random account IDs."""
    return [f"ACC{random.randint(100000, 999999)}" for _ in range(count)]

# Check if CSV has header
def csv_has_header(file_path: Path) -> bool:
    """Check if CSV file has a proper header row."""
    if not file_path.exists():
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            # Check if first line contains expected column names
            return 'account_id' in first_line and 'transaction_id' in first_line
    except:
        return False

# Generate transactions using LLM
def generate_transactions_llm(
    client: OpenAI,
    model: str,
    category_data: Dict,
    batch_size: int = 10,
    account_ids: List[str] = None
) -> List[Dict]:
    """Generate batch of transactions using OpenRouter LLM."""
    
    if account_ids is None:
        account_ids = generate_account_ids(20)
    
    # Construct prompt
    prompt = f"""Generate {batch_size} realistic UK SME bank transactions for this category.

Category: {category_data['name']}
Category Code: {category_data['category']}
Definition: {category_data['definition']}
Transaction Direction: {category_data['transaction_direction']} (CRITICAL: Must be exactly this)

Guidance for generating realistic descriptions:
{category_data['synthetic_prompt']}

CRITICAL RULES:
1. transaction_type MUST be "{category_data['transaction_direction']}" for ALL transactions
   - DEBIT means money OUT (negative impact on balance)
   - CREDIT means money IN (positive impact on balance)
2. amount must be a POSITIVE number (the sign is determined by transaction_type)
3. description must match realistic UK bank feed format
4. date must be in YYYY-MM-DD format within last 12 months
5. currency_code must be "GBP"

Return ONLY a valid JSON array with {batch_size} transactions, no other text.

Format:
[
  {{
    "description": "realistic UK bank description",
    "amount": positive_number,
    "transaction_type": "{category_data['transaction_direction']}",
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
                {"role": "system", "content": "You are a UK banking transaction data generator. Always return valid JSON arrays only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
            max_tokens=3000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean response (remove markdown code blocks if present)
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Parse JSON
        transactions = json.loads(content)
        
        # Add metadata to each transaction
        for i, txn in enumerate(transactions):
            txn['account_id'] = random.choice(account_ids)
            txn['transaction_id'] = f"TXN{random.randint(1000000, 9999999)}"
            txn['category'] = category_data['category']
            txn['primary_category'] = category_data['primary_category']
            txn['full_category_path'] = category_data['full_category_path']
            txn['confidence'] = 1.0  # Synthetic data has perfect confidence
            
            # Validate transaction_type matches category direction
            if category_data['transaction_direction'] != "BOTH":
                if txn['transaction_type'] != category_data['transaction_direction']:
                    print(f"WARNING: LLM generated wrong transaction_type. Correcting: {txn['transaction_type']} -> {category_data['transaction_direction']}")
                    txn['transaction_type'] = category_data['transaction_direction']
        
        return transactions
    
    except Exception as e:
        print(f"ERROR: Failed to generate transactions: {e}")
        return []

# Main generation function
def generate_synthetic_data(
    version: str,
    samples_per_category: int,
    batch_size: int = 10,
    config: Dict = None
):
    """Main function to generate synthetic transaction data."""
    
    print(f"\nStarting synthetic data generation for version: {version}")
    print("=" * 60)
    
    # Load category tree
    print("Loading category tree...")
    category_tree = load_category_tree(version)
    leaf_categories = extract_leaf_categories(category_tree)
    print(f"Found {len(leaf_categories)} leaf categories")
    
    # Load progress
    print("Loading progress tracker...")
    progress = load_progress(version)
    
    # Initialise OpenRouter client
    print("Connecting to OpenRouter...")
    client = init_openrouter_client(
        api_key=config['openrouter']['api_key'],
    )
    model = config['openrouter']['model']
    print(f"Using model: {model}")
    
    # Generate account IDs pool
    account_ids = generate_account_ids(20)
    print(f"Generated {len(account_ids)} account IDs")
    
    # Prepare output file
    output_path = Path(f"data/versions/{version}/synthetic_data.csv")
    
    # Check if file exists and has header
    has_header = csv_has_header(output_path)
    
    # CSV columns
    fieldnames = [
        'account_id', 'transaction_id', 'description', 'amount',
        'transaction_type', 'date', 'currency_code', 'category',
        'primary_category', 'full_category_path', 'confidence'
    ]
    
    # Open CSV file in append mode
    csv_file = open(output_path, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # Write header if file is new or doesn't have header
    if not has_header:
        writer.writeheader()
        print("Written CSV header")
    
    print("\n" + "=" * 60)
    print("Starting transaction generation...")
    print("=" * 60)
    
    total_generated = 0
    total_categories = len(leaf_categories)
    
    try:
        for idx, category_data in enumerate(leaf_categories, 1):
            category_code = category_data['category']
            category_name = category_data['name']
            
            # Check progress
            generated_so_far = progress.get(category_code, {}).get('generated', 0)
            
            if generated_so_far >= samples_per_category:
                print(f"[{idx}/{total_categories}] Skipping {category_name} (already complete: {generated_so_far}/{samples_per_category})")
                continue
            
            remaining = samples_per_category - generated_so_far
            
            print(f"\n[{idx}/{total_categories}] Generating for: {category_name}")
            print(f"  Category: {category_code}")
            print(f"  Direction: {category_data['transaction_direction']}")
            print(f"  Progress: {generated_so_far}/{samples_per_category}")
            print(f"  Remaining: {remaining}")
            
            category_generated = 0
            
            # Generate in batches
            while category_generated < remaining:
                current_batch_size = min(batch_size, remaining - category_generated)
                
                print(f"  Generating batch of {current_batch_size}...")
                
                transactions = generate_transactions_llm(
                    client=client,
                    model=model,
                    category_data=category_data,
                    batch_size=current_batch_size,
                    account_ids=account_ids
                )
                
                if not transactions:
                    print(f"  WARNING: Failed to generate batch, retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                
                # Write to CSV
                for txn in transactions:
                    writer.writerow(txn)
                
                category_generated += len(transactions)
                total_generated += len(transactions)
                
                # Update progress
                progress[category_code] = {
                    'category': category_name,
                    'generated': generated_so_far + category_generated,
                    'target': samples_per_category,
                    'completed': (generated_so_far + category_generated) >= samples_per_category,
                    'last_updated': datetime.now().isoformat()
                }
                save_progress(version, progress)
                
                print(f"  Generated {len(transactions)} transactions (total for category: {generated_so_far + category_generated}/{samples_per_category})")
                
                # Rate limiting
                time.sleep(0.5)
            
            print(f"  Completed {category_name}")
    
    finally:
        csv_file.close()
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total transactions generated this run: {total_generated}")
    print(f"Output file: {output_path}")
    print(f"Progress tracker: data/versions/{version}/generation_progress.json")
    
    # Summary stats
    completed = sum(1 for p in progress.values() if p.get('completed', False))
    print(f"\nSummary:")
    print(f"  Completed categories: {completed}/{total_categories}")
    print(f"  Total transactions in file: {sum(p.get('generated', 0) for p in progress.values())}")

# CLI
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic transaction data")
    parser.add_argument("--version", default="v1.0", help="Category tree version (default: v1.0)")
    parser.add_argument("--samples-per-category", type=int, default=100, help="Number of samples per category (default: 100)")
    parser.add_argument("--batch-size", type=int, default=10, help="Transactions per LLM call (default: 10)")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    # Validate OpenRouter API key
    if not config.get('openrouter', {}).get('api_key'):
        raise ValueError("OpenRouter API key not found in config/config.yaml. Please add it to the config file.")
    
    # Run generation
    generate_synthetic_data(
        version=args.version,
        samples_per_category=args.samples_per_category,
        batch_size=args.batch_size,
        config=config
    )

if __name__ == "__main__":
    main()