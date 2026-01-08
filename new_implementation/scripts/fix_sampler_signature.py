#!/usr/bin/env python3
"""Fix _get_train_sampler() method signature"""

import json
from pathlib import Path

notebook_path = Path('/home/hounfodjidagba/learning/cassava/new_implementation/notebooks/05_train_baseline_plus.ipynb')

print(f"Loading notebook: {notebook_path}")
with open(notebook_path, 'r') as f:
    nb = json.load(f)

print(f"Loaded {len(nb['cells'])} cells\n")

# Find and fix WeightedSamplerTrainer class
print("Fixing WeightedSamplerTrainer._get_train_sampler() signature...")
fixed = False
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'class WeightedSamplerTrainer' in source and '_get_train_sampler' in source:
            print(f"   Found at Cell {idx}")
            
            # Replace the entire cell with corrected version
            cell['source'] = [
                "from transformers import Trainer\n",
                "\n",
                "class WeightedSamplerTrainer(Trainer):\n",
                "    \"\"\"Custom Trainer with WeightedRandomSampler\"\"\"\n",
                "    \n",
                "    def __init__(self, *args, weighted_sampler=None, **kwargs):\n",
                "        super().__init__(*args, **kwargs)\n",
                "        self.weighted_sampler = weighted_sampler\n",
                "    \n",
                "    def _get_train_sampler(self, train_dataset):\n",
                "        \"\"\"Override to use weighted sampler. Note: must accept train_dataset parameter!\"\"\"\n",
                "        if self.weighted_sampler is not None:\n",
                "            return self.weighted_sampler\n",
                "        return super()._get_train_sampler(train_dataset)\n",
                "\n",
                "print(\"✅ WeightedSamplerTrainer class defined\")\n"
            ]
            fixed = True
            print(f"   ✅ Fixed method signature to accept train_dataset parameter")
            break

if not fixed:
    print("   ⚠️  Cell not found")

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"\n✅ Fixed! The method now accepts the dataset parameter correctly")
