"""Comprehensive fix for stage_b_dpo_training.py"""

# Read the file
with open('stage_b_dpo_training.py', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

# Fix each line
fixed_lines = []
for i, line in enumerate(lines, 1):
    # Fix the specific problem at line 593-594
    if i == 593:
        fixed_lines.append('    logger.info("\\nDPO Training completed successfully!")\n')
        continue
    elif i == 594:
        # Skip this line as it's merged into 593
        continue
    elif i == 600:
        # This is the trailing quote that needs to be removed
        continue
    
    # General fixes for escaped characters
    fixed_line = line
    fixed_lines.append(fixed_line)

# Write back
with open('stage_b_dpo_training.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print(f"✓ Fixed {len(fixed_lines)} lines")
print("✓ Fixed stage_b_dpo_training.py")
