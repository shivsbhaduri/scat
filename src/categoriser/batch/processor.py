"""
Batch CSV Processor

Processes CSV files containing transactions and outputs categorised results.
Supports incremental processing and error handling.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv
import time
from typing import List, Optional
from datetime import datetime

from categoriser.core.config import get_config
from categoriser.core.schemas import TransactionInput, CategorisationResult
from categoriser.engine.orchestrator import TransactionCategoriser


class CSVBatchProcessor:
    """
    Batch processor for CSV files containing transactions.
    """
    
    def __init__(self, version: Optional[str] = None):
        """
        Initialise the batch processor.
        
        Args:
            version: Model version to use (uses default if not specified)
        """
        self.config = get_config()
        self.version = version or self.config.default_version
        self.orchestrator = TransactionCategoriser(version=self.version)
    
    def _read_csv(self, input_path: Path) -> List[dict]:
        """
        Read transactions from CSV file.
        
        Args:
            input_path: Path to input CSV file
        
        Returns:
            List of transaction dictionaries
        
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If CSV format is invalid
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        transactions = []
        errors = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Required columns - more flexible now
            required_columns = [
                'description', 'amount', 'transaction_type'
            ]
            
            if not all(col in reader.fieldnames for col in required_columns):
                raise ValueError(
                    f"CSV missing required columns. Required: {required_columns}, "
                    f"Found: {reader.fieldnames}"
                )
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (after header)
                try:
                    # Parse transaction - use defaults for optional fields
                    transaction = {
                        'account_id': row.get('account_id', 'BATCH'),
                        'transaction_id': row.get('transaction_id', f'TXN_{row_num}'),
                        'description': row['description'],
                        'amount': float(row['amount']),
                        'transaction_type': row['transaction_type'].upper(),
                        'date': row.get('date', datetime.now().strftime('%Y-%m-%d')),
                        'currency_code': row.get('currency_code', 'GBP')
                    }
                    
                    # Validate with Pydantic
                    TransactionInput(**transaction)
                    transactions.append(transaction)
                
                except Exception as e:
                    errors.append({
                        'row': row_num,
                        'error': str(e),
                        'data': row
                    })
        
        if errors:
            print(f"WARNING: {len(errors)} rows failed validation:")
            for err in errors[:10]:  # Show first 10 errors
                print(f"  Row {err['row']}: {err['error']}")
            
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
        
        print(f"Loaded {len(transactions)} valid transactions from {input_path}")
        
        return transactions
    
    def _write_csv(
        self,
        results: List[dict],
        output_path: Path
    ):
        """
        Write categorisation results to CSV file.
        
        Args:
            results: List of result dictionaries
            output_path: Path to output CSV file
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Define output columns
        fieldnames = [
            'account_id',
            'transaction_id',
            'description',
            'amount',
            'transaction_type',
            'date',
            'currency_code',
            'category',
            'primary_category',
            'full_category_path',
            'confidence',
            'categorisation_method',
            'tier',
            'method',
            'processing_time_ms',
            'model_version'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            for result in results:
                writer.writerow(result)
        
        print(f"Written {len(results)} results to {output_path}")
    
    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None
    ) -> dict:
        """
        Process a CSV file and categorise all transactions.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to output CSV file (auto-generated if not provided)
        
        Returns:
            Dictionary with processing statistics
        """
        input_path = Path(input_path)
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{input_path.stem}_categorised_{timestamp}.csv"
            output_path = Path("data/batch/output") / output_filename
        else:
            output_path = Path(output_path)
        
        print(f"\nStarting batch processing")
        print("=" * 60)
        print(f"Input file: {input_path}")
        print(f"Output file: {output_path}")
        print(f"Model version: {self.version}")
        print("=" * 60)
        
        # Read input CSV
        start_time = time.time()
        transactions = self._read_csv(input_path)
        
        if not transactions:
            print("ERROR: No valid transactions found in input file")
            return {
                'success': False,
                'error': 'No valid transactions',
                'total_transactions': 0,
                'processing_time_seconds': 0
            }
        
        # Categorise transactions
        print(f"\nCategorising {len(transactions)} transactions...")
        categorisation_results = self.orchestrator.categorise_batch(transactions)
        
        # Merge original transaction data with categorisation results
        results = []
        for i, cat_result in enumerate(categorisation_results):
            # Map tier/method to categorisation_method
            tier = cat_result.get('tier', 'tier3')
            method = cat_result.get('method', 'tier3_xgboost')
            
            if tier == 'tier1':
                cat_method = 'EXACT_MATCH' if 'exact' in method else 'FUZZY_MATCH'
            elif tier == 'tier2':
                if 'keyword_direct' in method:
                    cat_method = 'SEMANTIC_KEYWORD'
                elif 'keyword' in method:
                    cat_method = 'SEMANTIC_KEYWORD'
                else:
                    cat_method = 'SEMANTIC_FULL'
            else:
                cat_method = 'HYBRID_XGB'
            
            # Extract category components
            category = cat_result['category']
            primary_category = category.split('_')[0] if category != 'UNKNOWN' else 'UNKNOWN'
            
            merged_result = {
                **transactions[i],  # Original transaction
                'category': category,
                'primary_category': primary_category,
                'full_category_path': category,
                'confidence': cat_result['confidence'],
                'categorisation_method': cat_method,
                'tier': tier,
                'method': method,
                'processing_time_ms': cat_result.get('processing_time_ms', 0),
                'model_version': self.version
            }
            results.append(merged_result)
        
        # Write output CSV
        self._write_csv(results, output_path)
        
        # Calculate statistics
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Tier distribution
        tier_counts = {}
        for result in results:
            tier = result['tier']
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        # Method distribution
        method_counts = {}
        for result in results:
            method = result['categorisation_method']
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Confidence distribution
        high_conf = sum(1 for r in results if r['confidence'] >= 0.90)
        med_conf = sum(1 for r in results if 0.75 <= r['confidence'] < 0.90)
        low_conf = sum(1 for r in results if r['confidence'] < 0.75)
        
        # Print statistics
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Total transactions: {len(results)}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Average time per transaction: {processing_time/len(results):.3f} seconds")
        
        print(f"\nTier breakdown:")
        for tier, count in sorted(tier_counts.items()):
            print(f"  {tier}: {count} ({count/len(results)*100:.1f}%)")
        
        print(f"\nCategorisation methods:")
        for method, count in sorted(method_counts.items()):
            print(f"  {method}: {count} ({count/len(results)*100:.1f}%)")
        
        print(f"\nConfidence distribution:")
        print(f"  High (>=0.90): {high_conf} ({high_conf/len(results)*100:.1f}%)")
        print(f"  Medium (0.75-0.90): {med_conf} ({med_conf/len(results)*100:.1f}%)")
        print(f"  Low (<0.75): {low_conf} ({low_conf/len(results)*100:.1f}%)")
        
        print(f"\nOutput file: {output_path}")
        
        # Print orchestrator statistics
        self.orchestrator.print_statistics()
        
        return {
            'success': True,
            'total_transactions': len(results),
            'processing_time_seconds': processing_time,
            'tier_distribution': tier_counts,
            'method_distribution': method_counts,
            'confidence_distribution': {
                'high': high_conf,
                'medium': med_conf,
                'low': low_conf
            },
            'output_path': str(output_path)
        }


def main():
    """CLI entry point for batch processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch categorise transactions from CSV")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument(
        "--output",
        help="Path to output CSV file (auto-generated if not provided)"
    )
    parser.add_argument(
        "--version",
        help="Model version to use (uses default if not specified)"
    )
    
    args = parser.parse_args()
    
    # Process file
    processor = CSVBatchProcessor(version=args.version)
    stats = processor.process_file(
        input_path=args.input_file,
        output_path=args.output
    )
    
    if not stats['success']:
        exit(1)


if __name__ == "__main__":
    main()