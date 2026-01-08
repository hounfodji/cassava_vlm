#!/usr/bin/env python3
"""Add weighted sampling and Trainer modifications"""

import json
from pathlib import Path

notebook_path = Path('/home/hounfodjidagba/learning/cassava/new_implementation/notebooks/05_train_baseline_plus.ipynb')

print(f"Loading notebook: {notebook_path}")
with open(notebook_path, 'r') as f:
    nb = json.load(f)

print(f"Loaded {len(nb['cells'])} cells\n")

# Add weighted sampling markdown cell after dataset loading (around Cell 14)
print("1. Adding weighted sampling markdown...")
weighted_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6.5 Light Weighted Sampling (Option E)\n",
        "\n",
        "**Key Difference**: Light oversampling (2-3x) instead of aggressive (16x)!\n",
        "\n",
        "- CBB: 2.5x (was 16x in Phase 1/2)\n",
        "- CGM: 2.0x (was 8x in Phase 1/2)\n",
        "- CMD: 1.0x (NOT penalized!)\n"
    ]
}
nb['cells'].insert(16, weighted_markdown)
print("   ✅ Added at Cell 16\n")

# Add weighted sampling code
print("2. Adding weighted sampling code...")
weighted_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import torch\n",
        "from torch.utils.data import WeightedRandomSampler\n",
        "from collections import Counter\n",
        "\n",
        "def extract_class_from_conversations(feature):\n",
        "    conversations = feature.get('conversations', [])\n",
        "    for turn in conversations:\n",
        "        if turn.get('role') == 'assistant':\n",
        "            content = turn.get('content', '')\n",
        "            if 'Bacterial Blight' in content or 'CBB' in content:\n",
        "                return 0\n",
        "            elif 'Brown Streak' in content or 'CBSD' in content:\n",
        "                return 1\n",
        "            elif 'Green Mottle' in content or 'CGM' in content:\n",
        "                return 2\n",
        "            elif 'Mosaic' in content or 'CMD' in content:\n",
        "                return 3\n",
        "            elif 'Healthy' in content:\n",
        "                return 4\n",
        "    return -1\n",
        "\n",
        "train_labels = [extract_class_from_conversations(f) for f in train_dataset]\n",
        "class_counts = Counter(train_labels)\n",
        "\n",
        "print(\"Class distribution:\")\n",
        "for cls, count in sorted(class_counts.items()):\n",
        "    if cls >= 0:\n",
        "        print(f\"  Class {cls}: {count} samples\")\n",
        "\n",
        "# LIGHT weights (NOT 16x!)\n",
        "light_weights = {0: 2.5, 1: 1.0, 2: 2.0, 3: 1.0, 4: 1.0}\n",
        "print(\"\\nLight class weights: CBB=2.5x, CGM=2.0x, others=1.0x\")\n",
        "\n",
        "sample_weights = [light_weights.get(label, 1.0) for label in train_labels]\n",
        "weighted_sampler = WeightedRandomSampler(\n",
        "    weights=torch.DoubleTensor(sample_weights),\n",
        "    num_samples=len(sample_weights),\n",
        "    replacement=True\n",
        ")\n",
        "print(f\"✅ Weighted sampler created\")\n"
    ]
}
nb['cells'].insert(17, weighted_code)
print("   ✅ Added at Cell 17\n")

# Add custom Trainer class
print("3. Adding WeightedSamplerTrainer class...")
trainer_class = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from transformers import Trainer\n",
        "\n",
        "class WeightedSamplerTrainer(Trainer):\n",
        "    \"\"\"Custom Trainer with WeightedRandomSampler\"\"\"\n",
        "    \n",
        "    def __init__(self, *args, weighted_sampler=None, **kwargs):\n",
        "        super().__init__(*args, **kwargs)\n",
        "        self.weighted_sampler = weighted_sampler\n",
        "    \n",
        "    def _get_train_sampler(self):\n",
        "        if self.weighted_sampler is not None:\n",
        "            return self.weighted_sampler\n",
        "        return super()._get_train_sampler()\n",
        "\n",
        "print(\"✅ WeightedSamplerTrainer class defined\")\n"
    ]
}
nb['cells'].insert(19, trainer_class)
print("   ✅ Added at Cell 19\n")

# Find and modify Trainer initialization
print("4. Modifying Trainer initialization...")
trainer_modified = False
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'trainer = Trainer(' in source and 'TrainingArguments' in source:
            print(f"   Found at Cell {idx}")
            # Replace Trainer with WeightedSamplerTrainer
            cell['source'] = [line.replace('trainer = Trainer(', 'trainer = WeightedSamplerTrainer(')
                            for line in cell['source']]
            # Add weighted_sampler parameter before closing )
            new_source = []
            for line in cell['source']:
                if line.strip() == ')':
                    new_source.append('    weighted_sampler=weighted_sampler,\n')
                new_source.append(line)
            cell['source'] = new_source
            trainer_modified = True
            print(f"   ✅ Modified at Cell {idx}\n")
            break

if not trainer_modified:
    print("   ⚠️  Trainer not found, needs manual modification\n")

# Modify model.generate() calls
print("5. Modifying model.generate() calls...")
generate_count = 0
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'model.generate(' in source and 'logits_processor' not in source:
            print(f"   Found at Cell {idx}")
            # Add logits_processor parameter
            new_source = []
            for line in cell['source']:
                new_source.append(line)
                if '**inputs,' in line:
                    new_source.append('        logits_processor=LogitsProcessorList([constrained_processor]),\n')
            cell['source'] = new_source
            generate_count += 1
            print(f"   ✅ Modified Cell {idx}")

print(f"   Total modified: {generate_count}\n")

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"✅ All modifications complete!")
print(f"   Total cells: {len(nb['cells'])}")
print("\n✨ Notebook ready for Option E training!")
