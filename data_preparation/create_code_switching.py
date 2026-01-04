#!/usr/bin/env python3
"""
Create code-switching examples mixing French and English.
Simulates natural bilingual conversations for multilingual training.
"""

import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm

CODE_SWITCH_PATTERNS = [
    # Pattern 1: Start French, switch to English
    {
        'start_lang': 'fr',
        'switch_at': 0.5,  # Switch halfway through conversation
        'transition': "Can you explain the treatment in English?"
    },
    # Pattern 2: Start English, switch to French
    {
        'start_lang': 'en',
        'switch_at': 0.5,
        'transition': "Pouvez-vous expliquer le traitement en français?"
    },
    # Pattern 3: Mixed throughout
    {
        'start_lang': 'fr',
        'switch_at': 0.33,
        'transition': "Et en anglais?"
    },
    # Pattern 4: Ask in one language, answer in another
    {
        'start_lang': 'en',
        'switch_at': 0.25,
        'transition': "Répondez en français s'il vous plaît."
    }
]

def create_code_switching_sample(en_sample, fr_sample, pattern=None):
    """
    Create code-switching conversation from English and French pairs.

    Args:
        en_sample: English conversation sample
        fr_sample: Corresponding French translation
        pattern: Switching pattern dict (or None for random)

    Returns:
        Code-switched conversation dict
    """
    if pattern is None:
        pattern = random.choice(CODE_SWITCH_PATTERNS)

    # Get conversations
    en_convs = en_sample['conversations']
    fr_convs = fr_sample['conversations']

    # Ensure same length
    min_len = min(len(en_convs), len(fr_convs))
    en_convs = en_convs[:min_len]
    fr_convs = fr_convs[:min_len]

    # Create mixed conversation
    mixed_convs = []
    switch_point = int(len(en_convs) * pattern['switch_at'])

    if pattern['start_lang'] == 'fr':
        # Start with French
        for i in range(switch_point):
            mixed_convs.append(fr_convs[i])

        # Add transition
        if switch_point < len(en_convs):
            mixed_convs.append({
                "from": "user",
                "value": pattern['transition']
            })

        # Continue with English
        for i in range(switch_point, len(en_convs)):
            mixed_convs.append(en_convs[i])

    else:
        # Start with English
        for i in range(switch_point):
            mixed_convs.append(en_convs[i])

        # Add transition
        if switch_point < len(fr_convs):
            mixed_convs.append({
                "from": "user",
                "value": pattern['transition']
            })

        # Continue with French
        for i in range(switch_point, len(fr_convs)):
            mixed_convs.append(fr_convs[i])

    return {
        "id": f"code_switch_{en_sample.get('id', 'unknown')}",
        "conversations": mixed_convs,
        "metadata": {
            "type": "code_switching",
            "pattern": pattern['start_lang'] + "_to_" + ("en" if pattern['start_lang'] == 'fr' else 'fr'),
            "original_id_en": en_sample.get('id'),
            "original_id_fr": fr_sample.get('id')
        }
    }

def create_code_switching_dataset(english_file, french_file, output_file, n_samples=None):
    """
    Create code-switching dataset from English and French datasets.

    Args:
        english_file: English dataset JSON
        french_file: French dataset JSON
        output_file: Output code-switching JSON
        n_samples: Number of samples to create (None = all)
    """
    print(f"Loading English data from: {english_file}")
    with open(english_file, 'r', encoding='utf-8') as f:
        english_data = json.load(f)

    print(f"Loading French data from: {french_file}")
    with open(french_file, 'r', encoding='utf-8') as f:
        french_data = json.load(f)

    print(f"Loaded {len(english_data)} EN and {len(french_data)} FR samples")

    # Ensure we have matching pairs
    n_pairs = min(len(english_data), len(french_data))

    if n_samples is None or n_samples > n_pairs:
        n_samples = n_pairs

    print(f"Creating {n_samples} code-switching samples...")

    # Shuffle to get random pairs
    random.seed(42)
    indices = random.sample(range(n_pairs), n_samples)

    code_switching_data = []

    for idx in tqdm(indices, desc="Creating code-switching pairs"):
        en_sample = english_data[idx]
        fr_sample = french_data[idx]

        # Create code-switched version
        cs_sample = create_code_switching_sample(en_sample, fr_sample)
        code_switching_data.append(cs_sample)

    print(f"Created {len(code_switching_data)} code-switching samples")

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(code_switching_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {output_file}")

    # Statistics
    patterns = {}
    for sample in code_switching_data:
        pattern = sample['metadata']['pattern']
        patterns[pattern] = patterns.get(pattern, 0) + 1

    print("\n=== Code-Switching Patterns ===")
    for pattern, count in patterns.items():
        print(f"  {pattern}: {count} samples")

def main():
    parser = argparse.ArgumentParser(
        description="Create code-switching dataset from EN and FR pairs"
    )
    parser.add_argument(
        '--english',
        type=str,
        required=True,
        help='English dataset JSON file'
    )
    parser.add_argument(
        '--french',
        type=str,
        required=True,
        help='French dataset JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output code-switching JSON file'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        help='Number of samples to create (default: all available pairs)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    random.seed(args.seed)

    create_code_switching_dataset(
        english_file=args.english,
        french_file=args.french,
        output_file=args.output,
        n_samples=args.n_samples
    )

if __name__ == '__main__':
    main()
