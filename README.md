# AgriVista Qwen3-VL-8B Fine-Tuning Project

**Objectif** : Créer un assistant agricole multimodal multilingue capable de détecter les maladies des cultures, tenir des conversations multi-tours, et supporter français, anglais et langues locales africaines.

## 📊 Stratégie de Fine-Tuning

**Approche** : 4-stage sequential fine-tuning avec replay buffers (35-48h GPU total)

1. **Stage 1** (6-8h) : Disease Recognition Foundation - PlantVillage 54k + CDDM diagnosis
2. **Stage 2** (18-24h) : Multi-Turn Conversations - CDDM 145k + replay buffer
3. **Stage 3** (3-4h) : High-Quality Instruction Tuning - Agri-LLaVA 6k
4. **Stage 4** (8-12h) : Multilingual Enhancement - French + local languages

## 🚀 Quick Start

### 1. Installation

```bash
# Clone le projet
cd /home/hounfodjidagba/learning/agrovista

# Installer les dépendances
pip install -r requirements.txt

# Setup Kaggle API pour PlantVillage
# Télécharger kaggle.json depuis https://www.kaggle.com/settings/account
mkdir -p ~/.kaggle
cp path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Télécharger les Datasets

```bash
# PlantVillage (54k images, ~2GB)
./scripts/download_plantvillage.sh

# CDDM est déjà disponible via symlink
# data/raw/cddm -> UnicomBenchmark/CDDMBench/dataset
```

### 3. Préparer les Données

```bash
# Créer PlantVillage VQA pairs
python data_preparation/create_plantvillage_vqa.py \
    --input_dir data/raw/plantvillage \
    --output_file data/stage1/plantvillage_vqa.json

# Formater CDDM pour Qwen3-VL
python data_preparation/format_to_qwen3vl.py --dataset cddm_simple
python data_preparation/format_to_qwen3vl.py --dataset cddm_full

# Valider les datasets
python data_preparation/validate_datasets.py data/stage1/plantvillage_vqa.json
```

### 4. Entraînement Stage 1

```bash
# Ouvrir le notebook Stage 1
jupyter notebook notebooks/stage1_disease_foundation.ipynb

# Ou lancer directement
jupyter nbconvert --to notebook --execute notebooks/stage1_disease_foundation.ipynb
```

## 📁 Structure du Projet

```
agrovista/
├── data/
│   ├── raw/                    # Datasets bruts
│   │   ├── plantvillage/       # PlantVillage 54k images
│   │   └── cddm/              # Symlink vers CDDM
│   ├── stage1/                 # Données Stage 1
│   ├── stage2/                 # Données Stage 2
│   ├── stage3/                 # Données Stage 3
│   └── stage4/                 # Données Stage 4 (multilingual)
│
├── data_preparation/           # Scripts de préparation
│   ├── create_plantvillage_vqa.py
│   ├── format_to_qwen3vl.py
│   ├── create_replay_buffer.py
│   ├── translate_to_french.py
│   ├── create_code_switching.py
│   └── validate_datasets.py
│
├── notebooks/                  # Notebooks d'entraînement
│   ├── stage1_disease_foundation.ipynb
│   ├── stage2_cddm_multiturn.ipynb
│   ├── stage3_agri_llava_quality.ipynb
│   ├── stage4_multilingual.ipynb
│   └── final_model_evaluation.ipynb
│
├── checkpoints/                # Checkpoints des modèles
│   ├── stage1_disease_foundation/
│   ├── stage2_cddm_multiturn/
│   ├── stage3_agri_llava_quality/
│   └── stage4_multilingual_final/
│
├── scripts/                    # Scripts utilitaires
│   └── download_plantvillage.sh
│
├── requirements.txt            # Dépendances Python
└── README.md                   # Ce fichier
```

## 🔧 Scripts de Préparation des Données

### create_plantvillage_vqa.py
Convertit le dataset PlantVillage en format VQA pour Qwen3-VL.

```bash
python data_preparation/create_plantvillage_vqa.py \
    --input_dir data/raw/plantvillage \
    --output_file data/stage1/plantvillage_vqa.json
```

### format_to_qwen3vl.py
Convertisseur universel pour CDDM, Agri-LLaVA, PlantVillage.

```bash
# CDDM simple (disease diagnosis)
python data_preparation/format_to_qwen3vl.py --dataset cddm_simple

# CDDM complet (145k conversations)
python data_preparation/format_to_qwen3vl.py --dataset cddm_full

# Agri-LLaVA
python data_preparation/format_to_qwen3vl.py --dataset agri_llava
```

### create_replay_buffer.py
Crée des replay buffers pour prévenir catastrophic forgetting.

```bash
# Replay buffer Stage 1 pour Stage 2
python data_preparation/create_replay_buffer.py \
    --source data/stage1/plantvillage_vqa.json \
    --output data/stage2/stage1_replay.json \
    --n_samples 10000 \
    --strategy stratified

# Replay buffer mixte (Stages 1+2+3) pour Stage 4
python data_preparation/create_replay_buffer.py \
    --sources stage1:data/stage1/plantvillage_vqa.json \
              stage2:data/stage2/cddm_full.json \
              stage3:data/stage3/agri_llava.json \
    --output data/stage4/english_replay.json \
    --n_samples 20000 \
    --strategy mixed_stages \
    --weights 0.3 0.5 0.2
```

### translate_to_french.py
Traduit les datasets en français via GPT-4 ou DeepL.

```bash
# Traduction GPT-4 (meilleure qualité)
export OPENAI_API_KEY="your-key"
python data_preparation/translate_to_french.py \
    --input data/stage2/cddm_full.json \
    --output data/stage4/cddm_french.json \
    --provider gpt4 \
    --n_samples 10000

# Traduction DeepL (plus rapide, moins cher)
export DEEPL_API_KEY="your-key"
python data_preparation/translate_to_french.py \
    --input data/stage1/plantvillage_vqa.json \
    --output data/stage4/plantvillage_french.json \
    --provider deepl \
    --n_samples 5000
```

### create_code_switching.py
Crée des conversations FR-EN mixtes pour multilinguisme naturel.

```bash
python data_preparation/create_code_switching.py \
    --english data/stage2/cddm_full.json \
    --french data/stage4/cddm_french.json \
    --output data/stage4/code_switching.json \
    --n_samples 10000
```

### validate_datasets.py
Valide le format et l'intégrité des datasets.

```bash
python data_preparation/validate_datasets.py data/stage1/plantvillage_vqa.json --verbose
```

## 📈 Métriques de Validation

### Stage 1 : Disease Recognition
- **Disease classification accuracy** : >90%
- **Crop identification accuracy** : >95%

### Stage 2 : Multi-Turn Conversations
- **Multi-turn coherence** : >0.85
- **Stage 1 retention** : >95%

### Stage 3 : Instruction Quality
- **Answer quality (GPT-4 judge)** : >4.0/5
- **Stage 2 retention** : >98%
- **Stage 1 retention** : >95%

### Stage 4 : Multilingual
- **French accuracy** : >85% of EN performance
- **English retention** : >98% (priorité absolue)
- **Local languages** : >70% basic comprehension

## 🔗 Ressources

- **Plan complet** : `.claude/plans/atomic-gliding-hippo.md`
- **CDDM Paper** : [ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11606.pdf)
- **Qwen3-VL Model** : [Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- **PlantVillage Dataset** : [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

## 🎯 Résultats Attendus

- ✅ **Disease detection** : 90-92% accuracy (EN)
- ✅ **Multi-turn dialogue** : 10+ tours cohérents
- ✅ **French support** : 85-87% de EN performance
- ✅ **Local languages** : 70-75% vocabulaire basique
- ✅ **General knowledge** : 95%+ Qwen3-VL base retained

## 💰 Coûts Estimés

- **GPU** : 35-48h V100 (~$50-100 cloud)
- **Traduction API** : $50-100 (GPT-4 + DeepL)
- **Stockage** : 150GB
- **Total** : ~$100-200 + 3-4 semaines temps

## 📝 Licence

Ce projet utilise les datasets suivants avec leurs licences respectives :
- **CDDM** : [License ECCV 2024]
- **PlantVillage** : CC BY-SA 4.0
- **Qwen3-VL** : Apache 2.0

## 🤝 Contribution

Pour contribuer au projet, veuillez :
1. Fork le repository
2. Créer une branche feature
3. Commit vos changements
4. Push et créer une Pull Request

## 📧 Contact

Pour questions ou support : [votre-email]
