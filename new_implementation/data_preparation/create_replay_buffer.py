#!/usr/bin/env python3
"""
Create replay buffers from previous training stages.
Uses stratified sampling to preserve class distribution.
"""

import os
import json
import random
import argparse
from pathlib import Path
from collections import Counter
from tqdm import tqdm

def stratified_sample(data, n_samples, stratify_by='disease'):
    """
    Perform stratified sampling to preserve class distribution.

    Args:
        data: List of samples
        n_samples: Number of samples to select
        stratify_by: Field to stratify by ('disease', 'crop', or 'random')

    Returns:
        List of sampled data
    """
    if stratify_by == 'random' or n_samples >= len(data):
        # Random sampling or taking all
        if n_samples >= len(data):
            return data
        return random.sample(data, n_samples)

    # Extract stratification keys
    keys = []
    for sample in data:
        if stratify_by == 'disease':
            # Extract disease from metadata or conversations
            if 'metadata' in sample and 'disease' in sample['metadata']:
                key = sample['metadata']['disease']
            else:
                # Try to infer from conversations
                key = 'unknown'
        elif stratify_by == 'crop':
            if 'metadata' in sample and 'crop' in sample['metadata']:
                key = sample['metadata']['crop']
            else:
                key = 'unknown'
        else:
            key = 'default'

        keys.append(key)

    # Count classes
    class_counts = Counter(keys)
    total_samples = len(data)

    # Calculate samples per class (proportional)
    samples_per_class = {}
    for cls, count in class_counts.items():
        proportion = count / total_samples
        samples_per_class[cls] = max(1, int(n_samples * proportion))

    # Adjust if total doesn't match n_samples exactly
    total_allocated = sum(samples_per_class.values())
    if total_allocated < n_samples:
        # Add remaining to largest class
        largest_class = max(class_counts, key=class_counts.get)
        samples_per_class[largest_class] += (n_samples - total_allocated)
    elif total_allocated > n_samples:
        # Remove from largest class
        largest_class = max(class_counts, key=class_counts.get)
        samples_per_class[largest_class] -= (total_allocated - n_samples)

    # Sample from each class
    sampled = []
    class_samples = {cls: [] for cls in class_counts}

    for sample, key in zip(data, keys):
        class_samples[key].append(sample)

    for cls, n in samples_per_class.items():
        if cls in class_samples and len(class_samples[cls]) > 0:
            n = min(n, len(class_samples[cls]))
            sampled.extend(random.sample(class_samples[cls], n))

    return sampled

def mixed_stage_sampling(stage_files, n_samples, weights=None):
    """
    Sample from multiple stages with given weights.

    Args:
        stage_files: List of (stage_name, file_path) tuples
        n_samples: Total samples to select
        weights: Optional list of weights for each stage (default: equal)

    Returns:
        List of sampled data with stage labels
    """
    if weights is None:
        weights = [1.0] * len(stage_files)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Load all stages
    all_stages = []
    for (stage_name, file_path), weight in zip(stage_files, weights):
        print(f"Loading {stage_name} from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        n_from_stage = int(n_samples * weight)
        print(f"  Sampling {n_from_stage} from {stage_name} ({len(data)} available)")

        # Sample from this stage
        sampled = random.sample(data, min(n_from_stage, len(data)))

        # Add stage label
        for sample in sampled:
            sample['replay_source'] = stage_name

        all_stages.extend(sampled)

    # If we're short, sample more from largest stage
    if len(all_stages) < n_samples:
        deficit = n_samples - len(all_stages)
        print(f"Need {deficit} more samples, sampling from stages...")
        # Load largest stage again and sample more
        largest_stage = max(stage_files, key=lambda x: len(json.load(open(x[1]))))
        with open(largest_stage[1], 'r') as f:
            extra = random.sample(json.load(f), deficit)
        all_stages.extend(extra)

    return all_stages[:n_samples]

def create_replay_buffer(source_files, output_file, n_samples, strategy='stratified', weights=None):
    """
    Create replay buffer from source file(s).

    Args:
        source_files: Single file or list of (name, path) tuples for mixed strategy
        output_file: Output JSON file
        n_samples: Number of samples
        strategy: 'stratified', 'random', or 'mixed_stages'
        weights: Weights for mixed_stages strategy
    """
    if strategy == 'mixed_stages':
        # Multiple source files
        sampled_data = mixed_stage_sampling(source_files, n_samples, weights)
    else:
        # Single source file
        if isinstance(source_files, list):
            source_file = source_files[0][1]
        else:
            source_file = source_files

        print(f"Loading data from: {source_file}")
        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Loaded {len(data)} samples")

        if n_samples > len(data):
            print(f"Warning: Requested {n_samples} samples but only {len(data)} available")
            n_samples = len(data)

        # Sample according to strategy
        if strategy == 'stratified':
            sampled_data = stratified_sample(data, n_samples, stratify_by='disease')
        else:  # random
            sampled_data = random.sample(data, n_samples)

    print(f"Selected {len(sampled_data)} samples")

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved replay buffer to {output_file}")

    # Statistics
    if strategy == 'mixed_stages':
        print("\n=== Replay Buffer Statistics ===")
        sources = Counter(s.get('replay_source', 'unknown') for s in sampled_data)
        for source, count in sources.most_common():
            print(f"  {source}: {count} samples")

def main():
    parser = argparse.ArgumentParser(
        description="Create replay buffer from previous training stages"
    )
    parser.add_argument(
        '--source',
        type=str,
        help='Source file (for single-stage replay)'
    )
    parser.add_argument(
        '--sources',
        type=str,
        nargs='+',
        help='Multiple source files for mixed-stage replay (format: stage1:path1 stage2:path2)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output replay buffer file'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        required=True,
        help='Number of samples to include'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='stratified',
        choices=['stratified', 'random', 'mixed_stages'],
        help='Sampling strategy'
    )
    parser.add_argument(
        '--weights',
        type=float,
        nargs='+',
        help='Weights for mixed_stages strategy'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    random.seed(args.seed)

    # Parse sources
    if args.strategy == 'mixed_stages':
        if not args.sources:
            raise ValueError("--sources required for mixed_stages strategy")

        source_files = []
        for src in args.sources:
            if ':' in src:
                name, path = src.split(':', 1)
                source_files.append((name, path))
            else:
                raise ValueError(f"Invalid source format: {src}. Use stage_name:path")
    else:
        if not args.source:
            raise ValueError("--source required for single-stage replay")
        source_files = args.source

    create_replay_buffer(
        source_files=source_files,
        output_file=args.output,
        n_samples=args.n_samples,
        strategy=args.strategy,
        weights=args.weights
    )

if __name__ == '__main__':
    main()
