#!/usr/bin/env python3
"""Simpler script to modify baseline notebook for Option E"""

import json
from pathlib import Path

notebook_path = Path('/home/hounfodjidagba/learning/cassava/new_implementation/notebooks/05_train_baseline_plus.ipynb')

print(f"Loading notebook: {notebook_path}")
with open(notebook_path, 'r') as f:
    nb = json.load(f)

print(f"Loaded {len(nb['cells'])} cells\n")

# 1. Update title (Cell 0)
print("1. Updating title...")
nb['cells'][0]['source'] = [
    "# Option E: Baseline + Light Fixes\n",
    "\n",
    "**Goal**: Start from proven baseline (70%, 97% CMD) and add MINIMAL targeted fixes.\n",
    "\n",
    "## Changes from Baseline:\n",
    "1. ✅ REAL constrained generation (eliminate 0.08% hallucination)\n",
    "2. ✅ Light weighted sampling (2.5x CBB, 2.0x CGM - NOT 16x!)\n",
    "3. ✅ NO aggressive augmentation (keep baseline transforms)\n",
    "4. ✅ Same training config (3 epochs, 3e-5 LR)\n",
    "\n",
    "## Expected Results:\n",
    "- Overall: ~70% (maintain baseline)\n",
    "- CMD: >95% (maintain baseline excellence)\n",
    "- CBB/CGM: 5-10% (small improvement)\n",
    "- Hallucination: 0% (with real constrained generation)\n"
]
print("   ✅ Title updated\n")

# 2. Add DISEASE_NAMES after imports (insert after Cell 2)
print("2. Adding DISEASE_NAMES constant...")
disease_names_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Valid disease names for constrained generation\n",
        "DISEASE_NAMES = [\n",
        "    \"Cassava Bacterial Blight (CBB)\",\n",
        "    \"Cassava Brown Streak Disease (CBSD)\",\n",
        "    \"Cassava Green Mottle (CGM)\",\n",
        "    \"Cassava Mosaic Disease (CMD)\",\n",
        "    \"Healthy\"\n",
        "]\n",
        "print(f\"✅ {len(DISEASE_NAMES)} valid disease classes defined\")\n"
    ]
}
nb['cells'].insert(3, disease_names_cell)
print("   ✅ Added at Cell 3\n")

# 3. Add RealConstrainedLogitsProcessor (insert after Cell 3)
print("3. Adding RealConstrainedLogitsProcessor class...")
processor_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from transformers import LogitsProcessor, LogitsProcessorList\n",
        "import torch\n",
        "\n",
        "class RealConstrainedLogitsProcessor(LogitsProcessor):\n",
        "    \"\"\"REAL constrained generation - forces output to valid disease names\"\"\"\n",
        "    \n",
        "    def __init__(self, processor, valid_diseases):\n",
        "        self.tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor\n",
        "        self.valid_diseases = valid_diseases\n",
        "        \n",
        "        # Tokenize \"Disease: \" prefix\n",
        "        self.disease_prefix_ids = self.tokenizer.encode(\"Disease: \", add_special_tokens=False)\n",
        "        \n",
        "        # Tokenize each disease name\n",
        "        self.disease_token_seqs = {}\n",
        "        for disease in valid_diseases:\n",
        "            tokens = self.tokenizer.encode(disease, add_special_tokens=False)\n",
        "            self.disease_token_seqs[disease] = tokens\n",
        "        \n",
        "        # Get first tokens\n",
        "        self.valid_first_tokens = set()\n",
        "        for tokens in self.disease_token_seqs.values():\n",
        "            if tokens:\n",
        "                self.valid_first_tokens.add(tokens[0])\n",
        "        \n",
        "        print(f\"✅ Constrained processor initialized with {len(valid_diseases)} classes\")\n",
        "    \n",
        "    def __call__(self, input_ids, scores):\n",
        "        batch_size = input_ids.shape[0]\n",
        "        \n",
        "        for i in range(batch_size):\n",
        "            seq = input_ids[i].tolist()\n",
        "            \n",
        "            # Check if just generated \"Disease: \"\n",
        "            if len(seq) >= len(self.disease_prefix_ids):\n",
        "                recent_tokens = seq[-len(self.disease_prefix_ids):]\n",
        "                \n",
        "                if recent_tokens == self.disease_prefix_ids:\n",
        "                    # Force next token to be from valid diseases\n",
        "                    scores[i, :] = float('-inf')\n",
        "                    for token_id in self.valid_first_tokens:\n",
        "                        scores[i, token_id] = 0.0\n",
        "        \n",
        "        return scores\n",
        "\n",
        "print(\"✅ RealConstrainedLogitsProcessor class defined\")\n"
    ]
}
nb['cells'].insert(4, processor_cell)
print("   ✅ Added at Cell 4\n")

# 4. Add processor initialization (insert after model loading - around Cell 8)
print("4. Adding processor initialization...")
init_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Initialize constrained processor\n",
        "constrained_processor = RealConstrainedLogitsProcessor(\n",
        "    processor=processor,\n",
        "    valid_diseases=DISEASE_NAMES\n",
        ")\n"
    ]
}
nb['cells'].insert(9, init_cell)
print("   ✅ Added at Cell 9\n")

print(f"✅ Modified notebook saved to {notebook_path}")
print(f"   Total cells: {len(nb['cells'])}")

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("\n✨ Option E modifications complete!")
print("Next: Add weighted sampling and modify Trainer manually in notebook")
