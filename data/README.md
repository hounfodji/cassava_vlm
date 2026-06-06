# Data Directory

Ce dossier contient tous les datasets pour le fine-tuning AgriVista Qwen3-VL-8B.

## Structure

```
data/
├── raw/                        # Datasets bruts (non versionnés)
│   ├── plantvillage/           # PlantVillage 54k images (~2GB)
│   ├── cddm/                   # Symlink vers CDDM dataset
│   └── agri_llava/             # Agri-LLaVA dataset
│
├── stage1/                     # Données Stage 1 : Disease Foundation
│   ├── plantvillage_vqa.json  # PlantVillage converti en VQA
│   ├── disease_diagnosis.json # CDDM simple Q&A
│   └── disease_knowledge.json # Connaissances expertes
│
├── stage2/                     # Données Stage 2 : Multi-Turn CDDM
│   ├── cddm_full.json         # 145k conversations CDDM
│   └── stage1_replay.json     # 10k replay buffer de Stage 1
│
├── stage3/                     # Données Stage 3 : Quality Tuning
│   ├── agri_llava.json        # 6k conversations Agri-LLaVA
│   ├── cddm_replay.json       # 1k replay CDDM
│   └── stage1_replay.json     # 500 replay disease accuracy
│
└── stage4/                     # Données Stage 4 : Multilingual
    ├── cddm_french.json       # 10k conversations FR (GPT-4)
    ├── plantvillage_french.json  # 5k VQA FR (DeepL)
    ├── code_switching.json    # 10k conversations FR-EN mixtes
    ├── bambara_seeds.json     # 500 seeds Bambara
    ├── wolof_seeds.json       # 500 seeds Wolof
    ├── yoruba_seeds.json      # 500 seeds Yoruba
    └── english_replay.json    # 20k replay EN (PRIORITÉ)
```

## Tailles Estimées

- **raw/plantvillage/** : ~2GB (54,000 images)
- **raw/cddm/** : ~5GB (145k conversations + images)
- **raw/agri_llava/** : ~3GB (6k conversations + images)
- **stage1/** : ~500MB
- **stage2/** : ~3GB
- **stage3/** : ~1GB
- **stage4/** : ~2GB

**Total** : ~17GB

## Download Instructions

### PlantVillage
```bash
./scripts/download_plantvillage.sh
```

### CDDM
Déjà disponible via symlink :
```bash
ln -s UnicomBenchmark/CDDMBench/dataset data/raw/cddm
```

### Agri-LLaVA
```bash
# Téléchargé automatiquement par les notebooks via Hugging Face Hub
# ou télécharger manuellement :
huggingface-cli download Agri-LLaVA-Anonymous/Agricultural_pests_and_diseases_instruction_tuning_data
```

## Processing Pipeline

1. **Stage 1 Data Preparation**
```bash
python data_preparation/create_plantvillage_vqa.py
python data_preparation/format_to_qwen3vl.py --dataset cddm_simple
```

2. **Stage 2 Data Preparation**
```bash
python data_preparation/format_to_qwen3vl.py --dataset cddm_full
python data_preparation/create_replay_buffer.py --source data/stage1/... --output data/stage2/stage1_replay.json --n_samples 10000
```

3. **Stage 3 Data Preparation**
```bash
python data_preparation/format_to_qwen3vl.py --dataset agri_llava
# Create replay buffers from Stage 1 and 2
```

4. **Stage 4 Data Preparation**
```bash
python data_preparation/translate_to_french.py --input data/stage2/cddm_full.json --output data/stage4/cddm_french.json --provider gpt4
python data_preparation/create_code_switching.py --english ... --french ... --output data/stage4/code_switching.json
```

## Validation

Valider chaque dataset avant entraînement :
```bash
python data_preparation/validate_datasets.py data/stage1/plantvillage_vqa.json --verbose
```

## Notes

- Les fichiers `*_checkpoint.json` sont créés pendant la traduction pour reprendre en cas d'interruption
- Ne pas versionner les datasets (trop volumineux) - seulement les scripts de génération
- Sauvegarder les datasets finaux sur stockage externe ou cloud
