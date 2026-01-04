#!/usr/bin/env python3
"""
Universal dataset converter to Qwen3-VL format.
Handles CDDM, Agri-LLaVA, and PlantVillage formats.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def convert_cddm_to_qwen3vl(sample):
    """
    Convert CDDM format to Qwen3-VL format.

    CDDM format already uses 'user'/'assistant' roles and <img> tags,
    so mainly just validation and standardization.
    """
    conversations = []

    for turn in sample['conversations']:
        role = turn.get('from', 'user')
        if role not in ['user', 'assistant']:
            role = 'user' if role == 'human' else 'assistant'

        content = turn.get('value', '')

        conversations.append({
            "from": role,
            "value": content
        })

    return {
        "id": sample.get('id', 'unknown'),
        "conversations": conversations
    }

def convert_agri_llava_to_qwen3vl(sample, image_dir):
    """
    Convert Agri-LLaVA format to Qwen3-VL format.

    Agri-LLaVA uses:
    - 'human'/'gpt' roles → convert to 'user'/'assistant'
    - '<image>' placeholder → convert to '<img>path</img>'
    """
    conversations = []

    # Get image path if available
    image_path = None
    if 'image' in sample:
        image_path = sample['image']
        if not image_path.startswith('/') and not image_path.startswith('data/'):
            # Make relative path
            image_path = os.path.join(image_dir, image_path)

    for i, turn in enumerate(sample['conversations']):
        role = turn.get('from', 'human')
        # Convert roles
        role = 'user' if role == 'human' else 'assistant'

        content = turn.get('value', '')

        # Replace <image> placeholder with actual path in first user turn
        if role == 'user' and i == 0 and '<image>' in content and image_path:
            content = content.replace('<image>', f'<img>{image_path}</img>')
        elif '<image>' in content:
            # Remove standalone <image> tags
            content = content.replace('<image>', '').strip()

        conversations.append({
            "from": role,
            "value": content
        })

    return {
        "id": sample.get('id', 'unknown'),
        "conversations": conversations
    }

def convert_vqa_test_to_qwen3vl(sample, image_base_dir='data/raw/cddm'):
    """
    Convert VQA test format (question/image/answer) to Qwen3-VL format.

    Used for disease_diagnosis.json which has:
    - question_id, question, image, answer fields
    """
    # Get image path and make it relative to project root if needed
    image_path = sample.get('image', '')

    # Handle various image path formats
    if image_path.startswith('/dataset/'):
        # /dataset/images/... -> data/raw/cddm/images/...
        image_path = image_path.replace('/dataset/', f'{image_base_dir}/')
    elif image_path.startswith('/home/jovyan/'):
        # Absolute paths from CDDM - convert to relative
        if '/images/' in image_path:
            rel_path = image_path.split('/images/')[-1]
            image_path = f'{image_base_dir}/images/{rel_path}'
    elif not image_path.startswith('data/'):
        # Relative paths
        image_path = f'{image_base_dir}/{image_path}'

    conversations = [
        {
            "from": "user",
            "value": f"<img>{image_path}</img>\n{sample.get('question', '')}"
        },
        {
            "from": "assistant",
            "value": sample.get('answer', '')
        }
    ]

    return {
        "id": sample.get('question_id', 'unknown'),
        "conversations": conversations
    }

def convert_plantvillage_to_qwen3vl(sample):
    """
    PlantVillage VQA is already in correct format from create_plantvillage_vqa.py.
    Just pass through with validation.
    """
    return {
        "id": sample.get('id', 'unknown'),
        "conversations": sample.get('conversations', [])
    }

def format_dataset(input_file, output_file, dataset_type, image_dir=None):
    """
    Format dataset to Qwen3-VL standard.

    Args:
        input_file: Input JSON file path
        output_file: Output JSON file path
        dataset_type: 'cddm', 'agri_llava', or 'plantvillage'
        image_dir: Base directory for images (for agri_llava)
    """
    print(f"Loading dataset from: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")
    print(f"Converting format: {dataset_type} → Qwen3-VL")

    converted_data = []

    for sample in tqdm(data, desc="Converting"):
        try:
            if dataset_type == 'cddm':
                converted = convert_cddm_to_qwen3vl(sample)
            elif dataset_type == 'cddm_vqa':
                # VQA test format (disease_diagnosis.json)
                converted = convert_vqa_test_to_qwen3vl(sample, image_dir or 'data/raw/cddm')
            elif dataset_type == 'agri_llava':
                converted = convert_agri_llava_to_qwen3vl(sample, image_dir or '')
            elif dataset_type == 'plantvillage':
                converted = convert_plantvillage_to_qwen3vl(sample)
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")

            converted_data.append(converted)

        except Exception as e:
            print(f"Warning: Error converting sample {sample.get('id', 'unknown')}: {e}")
            continue

    print(f"Successfully converted {len(converted_data)} samples")

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {output_file}")

    # Validation
    print("\n=== Validation ===")
    with_images = sum(1 for s in converted_data
                     if any('<img>' in str(turn.get('value', ''))
                           for turn in s['conversations']))

    total_turns = sum(len(s['conversations']) for s in converted_data)
    avg_turns = total_turns / len(converted_data) if converted_data else 0

    print(f"Samples with images: {with_images}/{len(converted_data)}")
    print(f"Average conversation turns: {avg_turns:.2f}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert datasets to Qwen3-VL format"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['cddm', 'cddm_simple', 'cddm_full', 'cddm_vqa', 'agri_llava', 'plantvillage'],
        help='Dataset type to convert'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        help='Input JSON file (optional, defaults based on dataset type)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='Output JSON file (optional, defaults based on dataset type)'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        help='Base directory for images (for agri_llava)'
    )

    args = parser.parse_args()

    # Default paths based on dataset type
    default_paths = {
        'cddm_simple': (
            'data/raw/cddm/VQA/disease_diagnosis.json',
            'data/stage1/disease_diagnosis.json'
        ),
        'cddm_vqa': (
            'data/raw/cddm/VQA/disease_diagnosis.json',
            'data/stage1/disease_diagnosis.json'
        ),
        'cddm_full': (
            'data/raw/cddm/VQA/Crop_Disease_train_qwenvl.json',
            'data/stage2/cddm_full.json'
        ),
        'cddm': (
            'data/raw/cddm/VQA/Crop_Disease_train_qwenvl.json',
            'data/stage2/cddm_full.json'
        ),
        'agri_llava': (
            'data/raw/agri_llava/agri_llava_instruction_tuning.json',
            'data/stage3/agri_llava.json'
        ),
        'plantvillage': (
            'data/stage1/plantvillage_vqa.json',
            'data/stage1/plantvillage_vqa_formatted.json'
        )
    }

    input_file = args.input_file or default_paths[args.dataset][0]
    output_file = args.output_file or default_paths[args.dataset][1]

    # Determine dataset type for converter
    # Map cddm_simple to cddm_vqa for VQA test format
    if args.dataset == 'cddm_simple':
        dataset_type = 'cddm_vqa'
    else:
        dataset_type = args.dataset.replace('_full', '')

    format_dataset(
        input_file=input_file,
        output_file=output_file,
        dataset_type=dataset_type,
        image_dir=args.image_dir
    )

if __name__ == '__main__':
    main()
