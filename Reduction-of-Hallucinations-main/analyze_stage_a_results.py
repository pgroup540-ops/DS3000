"""Quick analysis of Stage A evaluation results"""
import pandas as pd

# Load results
df = pd.read_csv('evaluation_results_stage_a/evaluation_results.csv')

print('='*60)
print('STAGE A INFERENCE EVALUATION SUMMARY')
print('='*60)
print(f'Total examples evaluated: {len(df)}')

# Calculate statistics
lengths = df['generated_summary'].str.split().str.len()
print(f'\nGenerated Summary Statistics:')
print(f'  Average length: {lengths.mean():.1f} words')
print(f'  Min length: {lengths.min()} words')
print(f'  Max length: {lengths.max()} words')
print(f'  Median length: {lengths.median():.0f} words')
print(f'  Std Dev: {lengths.std():.1f} words')

# Show some examples
print('\n' + '='*60)
print('SAMPLE OUTPUTS (First 5)')
print('='*60)

for i in range(min(5, len(df))):
    print(f'\nExample {i+1}:')
    print(f'Clinical Note: {df.iloc[i]["clinical_note"][:150]}...')
    print(f'Generated: {df.iloc[i]["generated_summary"][:200]}...')
    print('-'*60)

print('\n' + '='*60)
print('NEXT STEPS:')
print('='*60)
print('1. Review evaluation_results_stage_a/evaluation_results.csv')
print('2. Manually assess 20-50 examples for factual accuracy')
print('3. Calculate hallucination rate')
print('4. If hallucination rate < 15%: Deploy model')
print('5. If hallucination rate > 15%: Proceed to Stage B (DPO training)')
print('='*60)
