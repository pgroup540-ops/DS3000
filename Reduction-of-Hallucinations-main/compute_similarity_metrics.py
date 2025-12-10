"""
Compute Text Similarity Metrics for Stage B Evaluation
=======================================================

Computes BLEU, ROUGE, and BERTScore to compare predicted summaries
against reference summaries.

Usage:
    python compute_similarity_metrics.py \
        --results_csv "evaluation_results_stage_b_checkpoint1300/evaluation_results.csv" \
        --output_dir "evaluation_results_stage_b_checkpoint1300"
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. BLEU scores will not be computed.")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("rouge-score not available. ROUGE scores will not be computed.")

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logger.warning("bert-score not available. BERTScore will not be computed.")


def compute_bleu_score(reference, prediction):
    """Compute BLEU score for a single prediction."""
    if not NLTK_AVAILABLE:
        return None
    
    try:
        # Tokenize
        ref_tokens = word_tokenize(reference.lower())
        pred_tokens = word_tokenize(prediction.lower())
        
        # Compute BLEU with smoothing
        smoothing = SmoothingFunction().method1
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
        return bleu
    except Exception as e:
        logger.warning(f"Error computing BLEU: {e}")
        return None


def compute_rouge_scores(reference, prediction):
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    if not ROUGE_AVAILABLE:
        return None, None, None
    
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        
        rouge1 = scores['rouge1'].fmeasure
        rouge2 = scores['rouge2'].fmeasure
        rougeL = scores['rougeL'].fmeasure
        
        return rouge1, rouge2, rougeL
    except Exception as e:
        logger.warning(f"Error computing ROUGE: {e}")
        return None, None, None


def compute_bert_scores(references, predictions):
    """Compute BERTScore for all predictions at once (more efficient)."""
    if not BERTSCORE_AVAILABLE:
        return None, None, None
    
    try:
        logger.info("Computing BERTScores (this may take a few minutes)...")
        P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
        return P.tolist(), R.tolist(), F1.tolist()
    except Exception as e:
        logger.warning(f"Error computing BERTScore: {e}")
        return None, None, None


def compute_metrics(results_csv, output_dir):
    """Compute all similarity metrics."""
    
    logger.info("=" * 70)
    logger.info("Computing Text Similarity Metrics")
    logger.info("=" * 70)
    logger.info(f"Results CSV: {results_csv}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    # Check available metrics
    logger.info("Available metrics:")
    logger.info(f"  BLEU: {'✓' if NLTK_AVAILABLE else '✗ (install nltk)'}")
    logger.info(f"  ROUGE: {'✓' if ROUGE_AVAILABLE else '✗ (install rouge-score)'}")
    logger.info(f"  BERTScore: {'✓' if BERTSCORE_AVAILABLE else '✗ (install bert-score)'}")
    logger.info("")
    
    if not any([NLTK_AVAILABLE, ROUGE_AVAILABLE, BERTSCORE_AVAILABLE]):
        logger.error("No metric libraries available. Please install:")
        logger.error("  pip install nltk rouge-score bert-score")
        return
    
    # Load results
    logger.info("Loading evaluation results...")
    df = pd.read_csv(results_csv)
    logger.info(f"✓ Loaded {len(df)} examples")
    logger.info("")
    
    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize results storage
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    # Compute BLEU and ROUGE per example
    logger.info("Computing per-example metrics...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        reference = str(row['reference_summary'])
        prediction = str(row['predicted_summary'])
        
        # BLEU
        if NLTK_AVAILABLE:
            bleu = compute_bleu_score(reference, prediction)
            bleu_scores.append(bleu)
        
        # ROUGE
        if ROUGE_AVAILABLE:
            r1, r2, rL = compute_rouge_scores(reference, prediction)
            rouge1_scores.append(r1)
            rouge2_scores.append(r2)
            rougeL_scores.append(rL)
    
    # Add scores to dataframe
    if NLTK_AVAILABLE:
        df['bleu'] = bleu_scores
    if ROUGE_AVAILABLE:
        df['rouge1'] = rouge1_scores
        df['rouge2'] = rouge2_scores
        df['rougeL'] = rougeL_scores
    
    # Compute BERTScore (batch processing)
    if BERTSCORE_AVAILABLE:
        references = df['reference_summary'].astype(str).tolist()
        predictions = df['predicted_summary'].astype(str).tolist()
        
        P, R, F1 = compute_bert_scores(references, predictions)
        if F1 is not None:
            df['bertscore_precision'] = P
            df['bertscore_recall'] = R
            df['bertscore_f1'] = F1
    
    # Save detailed results
    detailed_csv = output_path / "evaluation_results_with_metrics.csv"
    df.to_csv(detailed_csv, index=False)
    logger.info(f"✓ Saved detailed results to {detailed_csv}")
    logger.info("")
    
    # Compute aggregate statistics
    logger.info("=" * 70)
    logger.info("Aggregate Metrics")
    logger.info("=" * 70)
    
    metrics_summary = {}
    
    if NLTK_AVAILABLE and bleu_scores:
        valid_bleu = [s for s in bleu_scores if s is not None]
        if valid_bleu:
            metrics_summary['bleu'] = {
                'mean': float(np.mean(valid_bleu)),
                'std': float(np.std(valid_bleu)),
                'min': float(np.min(valid_bleu)),
                'max': float(np.max(valid_bleu)),
                'median': float(np.median(valid_bleu))
            }
            logger.info(f"BLEU Score:")
            logger.info(f"  Mean: {metrics_summary['bleu']['mean']:.4f}")
            logger.info(f"  Std:  {metrics_summary['bleu']['std']:.4f}")
            logger.info(f"  Min:  {metrics_summary['bleu']['min']:.4f}")
            logger.info(f"  Max:  {metrics_summary['bleu']['max']:.4f}")
            logger.info("")
    
    if ROUGE_AVAILABLE and rouge1_scores:
        for name, scores in [('rouge1', rouge1_scores), ('rouge2', rouge2_scores), ('rougeL', rougeL_scores)]:
            valid_scores = [s for s in scores if s is not None]
            if valid_scores:
                metrics_summary[name] = {
                    'mean': float(np.mean(valid_scores)),
                    'std': float(np.std(valid_scores)),
                    'min': float(np.min(valid_scores)),
                    'max': float(np.max(valid_scores)),
                    'median': float(np.median(valid_scores))
                }
                logger.info(f"{name.upper()} Score:")
                logger.info(f"  Mean: {metrics_summary[name]['mean']:.4f}")
                logger.info(f"  Std:  {metrics_summary[name]['std']:.4f}")
                logger.info(f"  Min:  {metrics_summary[name]['min']:.4f}")
                logger.info(f"  Max:  {metrics_summary[name]['max']:.4f}")
                logger.info("")
    
    if BERTSCORE_AVAILABLE and F1 is not None:
        metrics_summary['bertscore'] = {
            'precision_mean': float(np.mean(P)),
            'recall_mean': float(np.mean(R)),
            'f1_mean': float(np.mean(F1)),
            'f1_std': float(np.std(F1)),
            'f1_min': float(np.min(F1)),
            'f1_max': float(np.max(F1))
        }
        logger.info(f"BERTScore:")
        logger.info(f"  Precision: {metrics_summary['bertscore']['precision_mean']:.4f}")
        logger.info(f"  Recall:    {metrics_summary['bertscore']['recall_mean']:.4f}")
        logger.info(f"  F1:        {metrics_summary['bertscore']['f1_mean']:.4f}")
        logger.info(f"  F1 Std:    {metrics_summary['bertscore']['f1_std']:.4f}")
        logger.info("")
    
    # Breakdown by label
    logger.info("=" * 70)
    logger.info("Metrics by Label")
    logger.info("=" * 70)
    
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        logger.info(f"\n{label.upper()} examples (n={len(label_df)}):")
        
        if NLTK_AVAILABLE and 'bleu' in label_df.columns:
            valid_bleu = label_df['bleu'].dropna()
            if len(valid_bleu) > 0:
                logger.info(f"  BLEU: {valid_bleu.mean():.4f} ± {valid_bleu.std():.4f}")
        
        if ROUGE_AVAILABLE and 'rouge1' in label_df.columns:
            valid_rouge1 = label_df['rouge1'].dropna()
            valid_rouge2 = label_df['rouge2'].dropna()
            valid_rougeL = label_df['rougeL'].dropna()
            if len(valid_rouge1) > 0:
                logger.info(f"  ROUGE-1: {valid_rouge1.mean():.4f} ± {valid_rouge1.std():.4f}")
                logger.info(f"  ROUGE-2: {valid_rouge2.mean():.4f} ± {valid_rouge2.std():.4f}")
                logger.info(f"  ROUGE-L: {valid_rougeL.mean():.4f} ± {valid_rougeL.std():.4f}")
        
        if BERTSCORE_AVAILABLE and 'bertscore_f1' in label_df.columns:
            valid_bert = label_df['bertscore_f1'].dropna()
            if len(valid_bert) > 0:
                logger.info(f"  BERTScore F1: {valid_bert.mean():.4f} ± {valid_bert.std():.4f}")
    
    logger.info("")
    
    # Save summary
    summary_path = output_path / "similarity_metrics_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"✓ Saved metrics summary to {summary_path}")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("Metric Computation Complete!")
    logger.info("=" * 70)
    logger.info(f"Detailed results: {detailed_csv}")
    logger.info(f"Summary: {summary_path}")
    logger.info("=" * 70)
    
    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="Compute text similarity metrics")
    parser.add_argument("--results_csv", type=str, required=True,
                       help="Path to evaluation results CSV")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for metrics")
    
    args = parser.parse_args()
    
    compute_metrics(args.results_csv, args.output_dir)


if __name__ == "__main__":
    main()
