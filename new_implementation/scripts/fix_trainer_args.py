#!/usr/bin/env python3
"""Fix TrainingArguments - remove weighted_sampler parameter"""

import json
from pathlib import Path

notebook_path = Path('/home/hounfodjidagba/learning/cassava/new_implementation/notebooks/05_train_baseline_plus.ipynb')

print(f"Loading notebook: {notebook_path}")
with open(notebook_path, 'r') as f:
    nb = json.load(f)

print(f"Loaded {len(nb['cells'])} cells\n")

# Find the cell with TrainingArguments and remove weighted_sampler parameter
print("Fixing TrainingArguments...")
fixed = False
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'TrainingArguments(' in source and 'weighted_sampler=weighted_sampler,' in source:
            print(f"   Found at Cell {idx}")
            # Remove the weighted_sampler line
            cell['source'] = [line for line in cell['source'] 
                            if 'weighted_sampler=weighted_sampler,' not in line]
            fixed = True
            print(f"   ✅ Removed weighted_sampler from TrainingArguments")
            break

if not fixed:
    print("   ⚠️  Cell not found")

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\n✅ Fixed! Now weighted_sampler is only passed to WeightedSamplerTrainer")
