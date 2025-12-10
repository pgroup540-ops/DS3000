"""
Manual Assessment Tool for Stage A Evaluation
==============================================

This tool helps you manually review and assess model outputs for hallucinations.

Usage:
    python manual_assessment_tool.py

You'll review each example and mark if it contains hallucinations.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

class ManualAssessmentTool:
    def __init__(self, results_csv='evaluation_results_stage_a/evaluation_results.csv'):
        self.results_csv = results_csv
        self.df = pd.read_csv(results_csv)
        self.assessments = []
        self.assessment_file = 'evaluation_results_stage_a/manual_assessment.json'
        
        # Load existing assessments if any
        if Path(self.assessment_file).exists():
            with open(self.assessment_file, 'r') as f:
                saved = json.load(f)
                self.assessments = saved.get('assessments', [])
                print(f"✓ Loaded {len(self.assessments)} existing assessments\n")
    
    def run_assessment(self, start_idx=0, max_examples=50):
        """Run interactive assessment session."""
        print("="*70)
        print("MANUAL HALLUCINATION ASSESSMENT")
        print("="*70)
        print("\nInstructions:")
        print("  - Review the Clinical Note and Generated Summary")
        print("  - Determine if the summary contains hallucinations")
        print("  - A hallucination is information NOT supported by the clinical note")
        print("\nCommands:")
        print("  'y' or 'yes'  = Contains hallucination")
        print("  'n' or 'no'   = No hallucination (factually accurate)")
        print("  's' or 'skip' = Skip this example")
        print("  'q' or 'quit' = Save and quit")
        print("  'stats'       = Show current statistics")
        print("="*70)
        input("\nPress Enter to start assessment...")
        
        # Get already assessed IDs
        assessed_ids = {a['example_id'] for a in self.assessments}
        
        for idx in range(start_idx, min(len(self.df), start_idx + max_examples)):
            example_id = self.df.iloc[idx]['example_id']
            
            # Skip if already assessed
            if example_id in assessed_ids:
                continue
            
            clinical_note = self.df.iloc[idx]['clinical_note']
            generated = self.df.iloc[idx]['generated_summary']
            
            print("\n" + "="*70)
            print(f"Example {idx + 1}/{len(self.df)} (ID: {example_id})")
            print("="*70)
            print(f"\n📋 CLINICAL NOTE:")
            print("-"*70)
            print(clinical_note)
            print("\n" + "-"*70)
            print(f"\n🤖 GENERATED SUMMARY:")
            print("-"*70)
            print(generated)
            print("\n" + "="*70)
            
            while True:
                response = input("\nContains hallucination? (y/n/s/q/stats): ").strip().lower()
                
                if response == 'q' or response == 'quit':
                    self._save_assessments()
                    self._show_final_stats()
                    return
                
                elif response == 'stats':
                    self._show_current_stats()
                    continue
                
                elif response == 's' or response == 'skip':
                    print("⏭️  Skipped")
                    break
                
                elif response in ['y', 'yes', 'n', 'no']:
                    has_hallucination = response in ['y', 'yes']
                    
                    # Optional: Ask for hallucination type
                    hallucination_type = None
                    if has_hallucination:
                        print("\nHallucination type (optional, press Enter to skip):")
                        print("  1 = Fabricated information")
                        print("  2 = Incorrect inference")
                        print("  3 = Contradicts clinical note")
                        print("  4 = Adds unsupported details")
                        type_input = input("Type (1-4 or Enter): ").strip()
                        
                        type_map = {
                            '1': 'fabricated',
                            '2': 'incorrect_inference',
                            '3': 'contradiction',
                            '4': 'unsupported_details'
                        }
                        hallucination_type = type_map.get(type_input, 'unspecified')
                    
                    # Save assessment
                    self.assessments.append({
                        'example_id': int(example_id),
                        'has_hallucination': has_hallucination,
                        'hallucination_type': hallucination_type,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    status = "❌ Hallucination detected" if has_hallucination else "✅ Factually accurate"
                    print(f"\n{status}")
                    
                    # Auto-save every 10 assessments
                    if len(self.assessments) % 10 == 0:
                        self._save_assessments()
                        print(f"💾 Auto-saved ({len(self.assessments)} assessments)")
                    
                    break
                
                else:
                    print("❓ Invalid input. Please enter y/n/s/q/stats")
        
        # Finished all examples
        print("\n" + "="*70)
        print("✅ Completed all examples!")
        print("="*70)
        self._save_assessments()
        self._show_final_stats()
    
    def _save_assessments(self):
        """Save assessments to JSON file."""
        output = {
            'total_assessed': len(self.assessments),
            'timestamp': datetime.now().isoformat(),
            'assessments': self.assessments
        }
        
        with open(self.assessment_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Saved to {self.assessment_file}")
    
    def _show_current_stats(self):
        """Show current assessment statistics."""
        if not self.assessments:
            print("\n📊 No assessments yet")
            return
        
        total = len(self.assessments)
        hallucinations = sum(1 for a in self.assessments if a['has_hallucination'])
        accurate = total - hallucinations
        hallucination_rate = (hallucinations / total * 100) if total > 0 else 0
        
        print("\n" + "="*70)
        print("📊 CURRENT STATISTICS")
        print("="*70)
        print(f"Total assessed: {total}")
        print(f"Hallucinations: {hallucinations} ({hallucination_rate:.1f}%)")
        print(f"Accurate: {accurate} ({100-hallucination_rate:.1f}%)")
        print("="*70)
    
    def _show_final_stats(self):
        """Show final assessment statistics and recommendation."""
        if not self.assessments:
            print("\n📊 No assessments completed")
            return
        
        total = len(self.assessments)
        hallucinations = sum(1 for a in self.assessments if a['has_hallucination'])
        accurate = total - hallucinations
        hallucination_rate = (hallucinations / total * 100) if total > 0 else 0
        
        print("\n" + "="*70)
        print("📊 FINAL ASSESSMENT RESULTS")
        print("="*70)
        print(f"Total examples assessed: {total}")
        print(f"Hallucinations found: {hallucinations}")
        print(f"Factually accurate: {accurate}")
        print(f"\n🎯 Hallucination Rate: {hallucination_rate:.1f}%")
        print("="*70)
        
        # Recommendation
        print("\n" + "="*70)
        print("💡 RECOMMENDATION")
        print("="*70)
        
        if hallucination_rate < 15:
            print("✅ Hallucination rate < 15%")
            print("➡️  RECOMMENDATION: Stage A model is ready for deployment!")
            print("    You can proceed to prepare the model for production use.")
        else:
            print("⚠️  Hallucination rate > 15%")
            print("➡️  RECOMMENDATION: Proceed to Stage B (DPO Training)")
            print("    Stage B will teach the model to prefer factual outputs.")
            print("\n    Next steps:")
            print("    1. Generate DPO training data: python generate_phase2_data.py")
            print("    2. Run Stage B training: python stage_b_dpo_training.py")
        
        print("="*70)
        
        # Save final report
        report_file = 'evaluation_results_stage_a/assessment_report.txt'
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MANUAL HALLUCINATION ASSESSMENT REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Examples Assessed: {total}\n")
            f.write(f"Hallucinations Found: {hallucinations}\n")
            f.write(f"Factually Accurate: {accurate}\n")
            f.write(f"Hallucination Rate: {hallucination_rate:.1f}%\n\n")
            
            if hallucination_rate < 15:
                f.write("RECOMMENDATION: Deploy Stage A model\n")
            else:
                f.write("RECOMMENDATION: Proceed to Stage B (DPO Training)\n")
        
        print(f"\n📄 Full report saved to {report_file}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manual Assessment Tool")
    parser.add_argument(
        '--results_csv',
        default='evaluation_results_stage_a/evaluation_results.csv',
        help='Path to evaluation results CSV'
    )
    parser.add_argument(
        '--start_idx',
        type=int,
        default=0,
        help='Start from this example index'
    )
    parser.add_argument(
        '--max_examples',
        type=int,
        default=50,
        help='Maximum examples to assess'
    )
    
    args = parser.parse_args()
    
    tool = ManualAssessmentTool(results_csv=args.results_csv)
    tool.run_assessment(start_idx=args.start_idx, max_examples=args.max_examples)


if __name__ == "__main__":
    main()
