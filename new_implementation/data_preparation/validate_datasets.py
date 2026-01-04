#!/usr/bin/env python3
"""
Validate dataset format and integrity for Qwen3-VL training.
"""

import json
import argparse
from pathlib import Path
from collections import Counter

def validate_conversation_format(sample, sample_idx):
    """
    Validate single conversation sample.

    Returns:
        (is_valid, errors)
    """
    errors = []

    # Check required fields
    if 'conversations' not in sample:
        errors.append(f"Missing 'conversations' field")
        return False, errors

    if not isinstance(sample['conversations'], list):
        errors.append(f"'conversations' must be a list")
        return False, errors

    if len(sample['conversations']) == 0:
        errors.append(f"Empty conversations list")
        return False, errors

    # Validate each turn
    for turn_idx, turn in enumerate(sample['conversations']):
        if 'from' not in turn:
            errors.append(f"Turn {turn_idx}: missing 'from' field")

        if 'value' not in turn:
            errors.append(f"Turn {turn_idx}: missing 'value' field")

        role = turn.get('from', '')
        if role not in ['user', 'assistant', 'system']:
            errors.append(f"Turn {turn_idx}: invalid role '{role}' (must be user/assistant/system)")

        # Check for malformed image tags
        value = turn.get('value', '')
        if '<img>' in value and '</img>' not in value:
            errors.append(f"Turn {turn_idx}: malformed image tag (missing </img>)")

    return len(errors) == 0, errors

def validate_dataset(input_file, verbose=False):
    """
    Validate entire dataset.

    Args:
        input_file: Path to JSON dataset
        verbose: Print detailed errors

    Returns:
        (is_valid, stats_dict)
    """
    print(f"Validating dataset: {input_file}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON: {e}")
        return False, {}
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return False, {}

    if not isinstance(data, list):
        print(f"✗ Dataset must be a list, got {type(data)}")
        return False, {}

    print(f"Loaded {len(data)} samples")

    # Validation statistics
    stats = {
        'total_samples': len(data),
        'valid_samples': 0,
        'invalid_samples': 0,
        'samples_with_images': 0,
        'total_turns': 0,
        'turn_distribution': Counter(),
        'role_distribution': Counter(),
        'errors_by_type': Counter()
    }

    invalid_samples = []

    # Validate each sample
    for idx, sample in enumerate(data):
        is_valid, errors = validate_conversation_format(sample, idx)

        if is_valid:
            stats['valid_samples'] += 1

            # Gather statistics
            convs = sample['conversations']
            stats['total_turns'] += len(convs)
            stats['turn_distribution'][len(convs)] += 1

            for turn in convs:
                stats['role_distribution'][turn.get('from', 'unknown')] += 1

                # Check for images
                if '<img>' in turn.get('value', ''):
                    stats['samples_with_images'] += 1
                    break  # Count sample only once

        else:
            stats['invalid_samples'] += 1
            invalid_samples.append((idx, sample.get('id', 'unknown'), errors))

            for error in errors:
                # Categorize error
                if 'missing' in error.lower():
                    stats['errors_by_type']['missing_field'] += 1
                elif 'role' in error.lower():
                    stats['errors_by_type']['invalid_role'] += 1
                elif 'image' in error.lower():
                    stats['errors_by_type']['malformed_image_tag'] += 1
                else:
                    stats['errors_by_type']['other'] += 1

    # Calculate averages
    if stats['valid_samples'] > 0:
        stats['avg_turns_per_sample'] = stats['total_turns'] / stats['valid_samples']
    else:
        stats['avg_turns_per_sample'] = 0

    # Print results
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)

    if stats['invalid_samples'] == 0:
        print(f"✓ All {stats['total_samples']} samples are VALID")
    else:
        print(f"✗ Found {stats['invalid_samples']} INVALID samples out of {stats['total_samples']}")

    print(f"\n--- Statistics ---")
    print(f"Valid samples: {stats['valid_samples']}")
    print(f"Samples with images: {stats['samples_with_images']}")
    print(f"Total conversation turns: {stats['total_turns']}")
    print(f"Average turns per sample: {stats['avg_turns_per_sample']:.2f}")

    print(f"\n--- Turn Distribution ---")
    for n_turns in sorted(stats['turn_distribution'].keys())[:10]:
        count = stats['turn_distribution'][n_turns]
        print(f"  {n_turns} turns: {count} samples")

    print(f"\n--- Role Distribution ---")
    for role, count in stats['role_distribution'].most_common():
        print(f"  {role}: {count}")

    if stats['invalid_samples'] > 0:
        print(f"\n--- Errors by Type ---")
        for error_type, count in stats['errors_by_type'].most_common():
            print(f"  {error_type}: {count}")

    # Print detailed errors if verbose
    if verbose and invalid_samples:
        print(f"\n--- Detailed Errors ---")
        for idx, sample_id, errors in invalid_samples[:10]:  # Show first 10
            print(f"\nSample {idx} (id={sample_id}):")
            for error in errors:
                print(f"  - {error}")

        if len(invalid_samples) > 10:
            print(f"\n... and {len(invalid_samples) - 10} more invalid samples")

    print("="*60)

    return stats['invalid_samples'] == 0, stats

def main():
    parser = argparse.ArgumentParser(
        description="Validate dataset format for Qwen3-VL training"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to dataset JSON file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed error messages'
    )

    args = parser.parse_args()

    is_valid, stats = validate_dataset(args.input_file, verbose=args.verbose)

    if is_valid:
        print("\n✓ Dataset is valid and ready for training!")
        exit(0)
    else:
        print("\n✗ Dataset has errors. Please fix before training.")
        exit(1)

if __name__ == '__main__':
    main()
