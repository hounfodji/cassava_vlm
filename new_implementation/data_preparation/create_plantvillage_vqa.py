#!/usr/bin/env python3
"""
Create VQA (Visual Question Answering) pairs from PlantVillage dataset.
Converts image classification dataset into conversational format for Qwen3-VL.
"""

import os
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Template questions variées pour éviter overfitting
QUESTIONS = [
    "What disease affects this plant?",
    "Can you identify the disease in this image?",
    "Is this plant healthy or diseased? If diseased, what is it?",
    "Describe the condition of this crop.",
    "What's wrong with this plant?",
    "Can you diagnose the problem with this leaf?",
    "What do you see in this plant image?",
    "Does this plant have any diseases? If so, which one?",
]

def parse_label(label_str):
    """
    Parse PlantVillage label format: "Tomato___Late_Blight"
    Returns: (crop, disease)
    """
    if '___' in label_str:
        crop, disease = label_str.split('___', 1)
    else:
        # Fallback for different formats
        parts = label_str.split('_')
        if len(parts) >= 2:
            crop = parts[0]
            disease = '_'.join(parts[1:])
        else:
            crop = label_str
            disease = "Unknown"

    # Clean up names
    crop = crop.replace('_', ' ').strip()
    disease = disease.replace('_', ' ').strip()

    return crop, disease

def create_vqa_pair(image_path, label, image_id):
    """
    Create a VQA conversation pair from image and label.

    Args:
        image_path: Path to image file
        label: PlantVillage label (e.g., "Tomato___Late_Blight")
        image_id: Unique ID for this sample

    Returns:
        dict: VQA conversation in Qwen3-VL format
    """
    crop, disease = parse_label(label)

    # Random question selection
    question = random.choice(QUESTIONS)

    # Generate appropriate answer
    if disease.lower() in ['healthy', 'no disease']:
        answer = f"This {crop.lower()} plant appears healthy with no visible disease symptoms."
    else:
        answer = f"This {crop.lower()} plant shows signs of {disease}."

    return {
        "id": f"plantvillage_{image_id}",
        "conversations": [
            {
                "from": "user",
                "value": f"Picture 1: <img>{image_path}</img>\n{question}"
            },
            {
                "from": "assistant",
                "value": answer
            }
        ],
        "metadata": {
            "source": "plantvillage",
            "crop": crop,
            "disease": disease,
            "label": label
        }
    }

def verify_image(image_path):
    """Verify image can be opened"""
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception as e:
        print(f"Warning: Cannot verify image {image_path}: {e}")
        return False

def process_plantvillage_dataset(input_dir, output_file, verify_images=True):
    """
    Process entire PlantVillage dataset into VQA format.

    Expected structure:
    input_dir/
        Tomato___Late_Blight/
            image1.jpg
            image2.jpg
        Apple___Apple_Scab/
            image3.jpg
        ...

    Args:
        input_dir: Path to PlantVillage dataset root
        output_file: Path to output JSON file
        verify_images: Whether to verify images can be opened
    """
    input_path = Path(input_dir)
    dataset = []

    print(f"Processing PlantVillage dataset from: {input_path}")

    # Get all class directories
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]

    if not class_dirs:
        raise ValueError(f"No class directories found in {input_path}")

    print(f"Found {len(class_dirs)} disease classes")

    total_images = 0
    skipped_images = 0

    # Process each class
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        label = class_dir.name

        # Get all images in this class
        image_files = list(class_dir.glob('*.jpg')) + \
                     list(class_dir.glob('*.jpeg')) + \
                     list(class_dir.glob('*.png')) + \
                     list(class_dir.glob('*.JPG'))

        for img_file in image_files:
            # Verify image if requested
            if verify_images and not verify_image(img_file):
                skipped_images += 1
                continue

            # Create relative path for portability
            relative_path = str(img_file.relative_to(input_path.parent))

            # Generate unique ID
            image_id = f"{label}_{img_file.stem}"

            # Create VQA pair
            vqa_sample = create_vqa_pair(relative_path, label, image_id)
            dataset.append(vqa_sample)
            total_images += 1

    print(f"\nProcessed {total_images} images successfully")
    if skipped_images > 0:
        print(f"Skipped {skipped_images} corrupted images")

    # Save dataset
    print(f"Saving to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved {len(dataset)} VQA pairs to {output_file}")

    # Print statistics
    print("\n=== Dataset Statistics ===")
    crops = {}
    diseases = {}
    for sample in dataset:
        crop = sample['metadata']['crop']
        disease = sample['metadata']['disease']
        crops[crop] = crops.get(crop, 0) + 1
        diseases[disease] = diseases.get(disease, 0) + 1

    print(f"Total samples: {len(dataset)}")
    print(f"Unique crops: {len(crops)}")
    print(f"Unique diseases: {len(diseases)}")
    print(f"\nTop 10 crops:")
    for crop, count in sorted(crops.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {crop}: {count}")
    print(f"\nTop 10 diseases:")
    for disease, count in sorted(diseases.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {disease}: {count}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert PlantVillage dataset to VQA format for Qwen3-VL"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='data/raw/plantvillage',
        help='Path to PlantVillage dataset directory'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='data/stage1/plantvillage_vqa.json',
        help='Path to output JSON file'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip image verification (faster but may include corrupted images)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Process dataset
    process_plantvillage_dataset(
        input_dir=args.input_dir,
        output_file=args.output_file,
        verify_images=not args.no_verify
    )

if __name__ == '__main__':
    main()
