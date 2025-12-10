"""Fix the escaped newlines in stage_b_dpo_training.py"""

# Read the broken file
with open('stage_b_dpo_training.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the literal \n escapes
fixed = content.replace(r'\n', '\n')
fixed = fixed.replace(r'\\', '\\')
fixed = fixed.replace(r'\"', '"')

# Write fixed version
with open('stage_b_dpo_training.py', 'w', encoding='utf-8') as f:
    f.write(fixed)

print("✓ Fixed stage_b_dpo_training.py")
