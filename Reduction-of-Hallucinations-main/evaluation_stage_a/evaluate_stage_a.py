"""
Stage A Model Evaluation
=========================

Evaluate the fine-tuned Stage A model for quality and hallucination rate.

Usage:
    python evaluate_stage_a.py \
        --model_path ./models/sft_specialist/final_model \
        --test_data phase1_data/sft/validation_set_processed.csv \
        --output_dir ./evaluation_results
"""

import os
import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime
from sft_inference import SFTInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate Stage A model quality."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """Initialize evaluator with model."""
        logger.info(f"Loading model from {model_path}")
        self.inference = SFTInference(
            model_path=model_path,
            device=device
        )
        self.results = []
    
    def evaluate_dataset(
        self,
        test_csv: str,
        max_examples: int = None
    ) -> Dict:
        """
        Evaluate model on test dataset.
        
        Args:
            test_csv: Path to test CSV with columns: clinical_note, summary
            max_examples: Limit number of examples (None = all)
        
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Loading test data from {test_csv}")
        df = pd.read_csv(test_csv)
        
        if max_examples:
            df = df.head(max_examples)
        
        logger.info(f"Evaluating {len(df)} examples...")
        
        for idx, row in df.iterrows():
            clinical_note = row['clinical_note']
            ground_truth = row.get('summary', row.get('truth', ''))
            
            # Generate summary
            result = self.inference.generate_summary(
                clinical_note,
                temperature=0.7,
                max_new_tokens=150
            )
            
            # Store result
            self.results.append({
                'example_id': idx,
                'clinical_note': clinical_note,
                'ground_truth': ground_truth,
                'generated_summary': result['generated_summary'],
            })
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} examples")
        
        logger.info("Evaluation complete!")
        return self._compute_stats()
    
    def _compute_stats(self) -> Dict:
        """Compute evaluation statistics."""
        stats = {
            'total_examples': len(self.results),
            'avg_generated_length': sum(
                len(r['generated_summary'].split()) for r in self.results
            ) / len(self.results),
            'timestamp': datetime.now().isoformat(),
        }
        return stats
    
    def save_results(self, output_dir: str):
        """Save evaluation results to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results CSV
        results_csv = output_path / "evaluation_results.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(results_csv, index=False)
        logger.info(f"Saved results to {results_csv}")
        
        # Save statistics
        stats = self._compute_stats()
        stats_json = output_path / "evaluation_stats.json"
        with open(stats_json, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {stats_json}")
        
        # Create human-readable report
        report_path = output_path / "evaluation_report.txt"
        self._create_report(report_path, stats)
        logger.info(f"Saved report to {report_path}")
    
    def _create_report(self, report_path: Path, stats: Dict):
        """Create human-readable evaluation report."""
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("STAGE A MODEL EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Evaluation Date: {stats['timestamp']}\n")
            f.write(f"Total Examples: {stats['total_examples']}\n")
            f.write(f"Avg Generated Length: {stats['avg_generated_length']:.1f} words\n\n")
            
            f.write("="*60 + "\n")
            f.write("SAMPLE OUTPUTS (First 10)\n")
            f.write("="*60 + "\n\n")
            
            for i, result in enumerate(self.results[:10], 1):
                f.write(f"Example {i}:\n")
                f.write(f"Clinical Note:\n{result['clinical_note']}\n\n")
                f.write(f"Ground Truth:\n{result['ground_truth']}\n\n")
                f.write(f"Generated:\n{result['generated_summary']}\n\n")
                f.write("-"*60 + "\n\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("NEXT STEPS:\n")
            f.write("="*60 + "\n")
            f.write("1. Review evaluation_results.csv for all outputs\n")
            f.write("2. Manually check 20-50 examples for hallucinations\n")
            f.write("3. Count hallucinations and calculate rate\n")
            f.write("4. Decision:\n")
            f.write("   - If hallucination rate <15%: Proceed to deployment\n")
            f.write("   - If hallucination rate >15%: Run Stage B (DPO)\n")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Stage A Model Evaluation")
    
    parser.add_argument(
        "--model_path",
        default="./models/sft_specialist/final_model",
        help="Path to trained model"
    )
    parser.add_argument(
        "--test_data",
        default="phase1_data/sft/validation_set_processed.csv",
        help="Path to test data CSV"
    )
    parser.add_argument(
        "--output_dir",
        default="./evaluation_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Max examples to evaluate (None = all)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        device=args.device
    )
    
    stats = evaluator.evaluate_dataset(
        test_csv=args.test_data,
        max_examples=args.max_examples
    )
    
    evaluator.save_results(args.output_dir)
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION COMPLETE!")
    logger.info("="*60)
    logger.info(f"\nResults saved to: {args.output_dir}")
    logger.info(f"Total examples: {stats['total_examples']}")
    logger.info("\nNext step: Review evaluation_results.csv manually")
    logger.info("Count hallucinations to determine if Stage B is needed\n")


if __name__ == "__main__":
    main()
