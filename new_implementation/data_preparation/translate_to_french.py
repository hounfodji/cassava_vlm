#!/usr/bin/env python3
"""
Translate agricultural datasets to French using GPT-4 or DeepL.
Preserves technical terminology and agricultural context.
"""

import os
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm

# Try importing OpenAI (GPT-4)
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai package not found. Install with: pip install openai")

# Try importing DeepL
try:
    import deepl
    HAS_DEEPL = True
except ImportError:
    HAS_DEEPL = False
    print("Warning: deepl package not found. Install with: pip install deepl")

TRANSLATION_PROMPT_GPT4 = """You are translating agricultural disease diagnosis conversations from English to French.

Requirements:
1. Use French agricultural terminology common in West Africa
2. Preserve disease names in their scientific/common form (e.g., "Late Blight" → "Mildiou")
3. Preserve crop names (e.g., "Tomato" → "Tomate", "Rice" → "Riz")
4. Maintain technical accuracy for symptoms, treatments, and recommendations
5. Use natural, conversational French (not overly formal)
6. Preserve JSON structure exactly

IMPORTANT: Translate ONLY the conversation content. Do NOT translate:
- Image paths (<img>...</img>)
- IDs
- Field names (from, value, etc.)

Return valid JSON with the same structure as input."""

def translate_with_gpt4(conversation, api_key, model="gpt-4"):
    """
    Translate conversation using OpenAI GPT-4.

    Args:
        conversation: dict with conversations to translate
        api_key: OpenAI API key
        model: Model to use (gpt-4 or gpt-4-turbo)

    Returns:
        Translated conversation dict
    """
    if not HAS_OPENAI:
        raise ImportError("openai package required. Install with: pip install openai")

    openai.api_key = api_key

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": TRANSLATION_PROMPT_GPT4},
                {"role": "user", "content": json.dumps(conversation, ensure_ascii=False)}
            ],
            temperature=0.3,  # Low temperature for consistency
            max_tokens=2000
        )

        # Parse response
        translated_text = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if translated_text.startswith('```json'):
            translated_text = translated_text.split('```json')[1]
        if translated_text.endswith('```'):
            translated_text = translated_text.rsplit('```', 1)[0]

        translated = json.loads(translated_text.strip())
        return translated

    except Exception as e:
        print(f"Error translating with GPT-4: {e}")
        return None

def translate_with_deepl(text, api_key, target_lang='FR'):
    """
    Translate text using DeepL API.

    Args:
        text: Text to translate
        api_key: DeepL API key
        target_lang: Target language code (FR for French)

    Returns:
        Translated text
    """
    if not HAS_DEEPL:
        raise ImportError("deepl package required. Install with: pip install deepl")

    translator = deepl.Translator(api_key)

    try:
        result = translator.translate_text(text, target_lang=target_lang)
        return result.text
    except Exception as e:
        print(f"Error translating with DeepL: {e}")
        return None

def translate_conversation_deepl(conversation, api_key):
    """
    Translate conversation using DeepL (simpler, no JSON parsing).

    Args:
        conversation: dict with conversations
        api_key: DeepL API key

    Returns:
        Translated conversation dict
    """
    translated = {
        "id": conversation['id'] + "_fr",
        "conversations": []
    }

    for turn in conversation['conversations']:
        value = turn['value']

        # Extract and preserve image tags
        img_tag = ''
        if '<img>' in value:
            # Extract image path
            start = value.find('<img>')
            end = value.find('</img>') + 6
            img_tag = value[start:end]
            # Remove from text to translate
            value_to_translate = value[:start] + value[end:]
        else:
            value_to_translate = value

        # Translate
        translated_text = translate_with_deepl(value_to_translate, api_key)

        if translated_text:
            # Re-add image tag at the beginning if it existed
            if img_tag:
                translated_text = img_tag + '\n' + translated_text

            translated['conversations'].append({
                "from": turn['from'],
                "value": translated_text
            })
        else:
            # Keep original if translation failed
            translated['conversations'].append(turn)

    return translated

def translate_dataset(input_file, output_file, provider='gpt4', api_key=None, n_samples=None, batch_size=10):
    """
    Translate entire dataset.

    Args:
        input_file: Input JSON file
        output_file: Output JSON file
        provider: 'gpt4' or 'deepl'
        api_key: API key for translation service
        n_samples: Number of samples to translate (None = all)
        batch_size: Save checkpoint every N samples
    """
    print(f"Loading dataset from: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    if n_samples:
        import random
        random.seed(42)
        data = random.sample(data, min(n_samples, len(data)))
        print(f"Sampling {len(data)} samples for translation")

    # Check for existing translations
    output_path = Path(output_file)
    checkpoint_file = output_path.parent / f"{output_path.stem}_checkpoint.json"

    translated_data = []
    start_idx = 0

    if checkpoint_file.exists():
        print(f"Found checkpoint at {checkpoint_file}")
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            translated_data = json.load(f)
        start_idx = len(translated_data)
        print(f"Resuming from sample {start_idx}")

    # Translate
    failed = 0
    for i, sample in enumerate(tqdm(data[start_idx:], desc=f"Translating ({provider})", initial=start_idx, total=len(data))):
        try:
            if provider == 'gpt4':
                translated = translate_with_gpt4(sample, api_key)
            elif provider == 'deepl':
                translated = translate_conversation_deepl(sample, api_key)
            else:
                raise ValueError(f"Unknown provider: {provider}")

            if translated:
                translated_data.append(translated)
            else:
                failed += 1
                print(f"Warning: Failed to translate sample {sample.get('id', 'unknown')}")

            # Save checkpoint periodically
            if (len(translated_data) % batch_size) == 0:
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(translated_data, f, indent=2, ensure_ascii=False)

            # Rate limiting
            if provider == 'gpt4':
                time.sleep(0.5)  # Avoid hitting rate limits

        except KeyboardInterrupt:
            print("\nInterrupted! Saving checkpoint...")
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(translated_data, f, indent=2, ensure_ascii=False)
            print(f"Checkpoint saved to {checkpoint_file}")
            return

        except Exception as e:
            print(f"Error translating sample {i}: {e}")
            failed += 1
            continue

    print(f"\nTranslation complete!")
    print(f"Successfully translated: {len(translated_data)}")
    print(f"Failed: {failed}")

    # Save final output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {output_file}")

    # Remove checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

def main():
    parser = argparse.ArgumentParser(
        description="Translate agricultural datasets to French"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file'
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='gpt4',
        choices=['gpt4', 'deepl'],
        help='Translation provider'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        help='API key (or set OPENAI_API_KEY / DEEPL_API_KEY env var)'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        help='Number of samples to translate (default: all)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='Save checkpoint every N samples'
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key
    if not api_key:
        if args.provider == 'gpt4':
            api_key = os.getenv('OPENAI_API_KEY')
        elif args.provider == 'deepl':
            api_key = os.getenv('DEEPL_API_KEY')

    if not api_key:
        raise ValueError(f"API key required. Set via --api_key or {args.provider.upper()}_API_KEY env var")

    translate_dataset(
        input_file=args.input,
        output_file=args.output,
        provider=args.provider,
        api_key=api_key,
        n_samples=args.n_samples,
        batch_size=args.batch_size
    )

if __name__ == '__main__':
    main()
