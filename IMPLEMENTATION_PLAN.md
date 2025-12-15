# Plan d'Implémentation CassavaVLM - De A à Z

## Vue d'ensemble

Implémentation complète d'un système Vision-Language Model pour le diagnostic interactif des maladies du manioc, basé sur:
- **Modèle**: Qwen2.5-VL-7B avec LoRA (r=16, α=32)
- **Prétraitement**: GroundingDINO + SAM-3 pour segmentation des feuilles
- **Support**: Français natif, déploiement hybride (cloud + edge)
- **Inspiration**: Architecture TLDVLM (97.27% sur tomate) adaptée au manioc

### Choix architecturaux validés
- ✅ Architecture modulaire professionnelle
- ✅ RTX 5090 32GB disponible localement (entraînement optimal)
- ✅ Notebooks Jupyter pour R&D → Scripts Python production
- ✅ RAG différé en Phase 2 (focus VLM d'abord)

---

## PHASE 1: Infrastructure et Environnement (Jours 1-3)

### 1.1 Structure du projet

Créer l'arborescence modulaire suivante:

```
cassava/
├── README.md                          # Documentation projet
├── .gitignore                         # Git ignore (data/, models/, checkpoints/)
├── requirements.txt                   # Dépendances Python
├── pyproject.toml                     # Configuration Poetry (optionnel)
├── setup.py                           # Installation package
│
├── configs/                           # ⭐ Configurations YAML
│   ├── model/
│   │   ├── qwen2_5_vl_7b.yaml        # Config Qwen2.5-VL-7B
│   │   ├── lora.yaml                  # Config LoRA (r=16, α=32)
│   │   └── preprocessing.yaml         # Config GroundingDINO + SAM-3
│   ├── training/
│   │   ├── phase1_projector.yaml     # Alignement projecteur
│   │   ├── phase2_lora.yaml          # Fine-tuning LoRA
│   │   └── hyperparameters.yaml      # LR, batch, epochs, etc.
│   └── deployment/
│       ├── cloud.yaml                 # Config API cloud
│       ├── edge_jetson.yaml          # Config Jetson Orin
│       └── mobile_flutter.yaml       # Config app mobile
│
├── data/                              # 📁 Données (gitignored)
│   ├── raw/                           # Datasets originaux
│   │   ├── kaggle_cassava_2020/
│   │   ├── iita_tanzania/
│   │   └── makerere_uganda/
│   ├── processed/                     # Images prétraitées
│   │   ├── segmented/                 # Feuilles segmentées
│   │   └── cropped/                   # Feuilles recadrées
│   ├── multimodal/                    # Paires image-texte
│   │   ├── train.json                 # Format LLaVA
│   │   ├── val.json
│   │   └── test.json
│   └── splits/
│       └── train_val_test_split.json  # Split stratifié 80/10/10
│
├── notebooks/                         # 📓 Expérimentation Jupyter
│   ├── 01_data_exploration.ipynb      # EDA + statistiques
│   ├── 02_preprocessing_pipeline.ipynb # GroundingDINO + SAM-3
│   ├── 03_data_generation.ipynb       # Génération paires multimodales
│   ├── 04_model_loading.ipynb         # Test chargement Qwen2.5-VL
│   ├── 05_lora_setup.ipynb            # Configuration LoRA
│   ├── 06_training_phase1.ipynb       # Entraînement projecteur
│   ├── 07_training_phase2.ipynb       # Fine-tuning LoRA
│   ├── 08_evaluation.ipynb            # Métriques + confusion matrix
│   └── 09_inference_demo.ipynb        # Démo interactive
│
├── src/                               # 📦 Code Python production
│   ├── __init__.py
│   │
│   ├── data/                          # Module données
│   │   ├── __init__.py
│   │   ├── downloaders.py             # Téléchargement Kaggle/IITA
│   │   ├── preprocessors.py           # Pipeline GroundingDINO + SAM-3
│   │   ├── augmentation.py            # Data augmentation
│   │   ├── multimodal_generator.py    # Génération paires image-texte
│   │   └── dataloader.py              # DataLoader PyTorch custom
│   │
│   ├── models/                        # Module modèles
│   │   ├── __init__.py
│   │   ├── qwen_vlm.py                # Wrapper Qwen2.5-VL-7B
│   │   ├── lora_config.py             # Configuration LoRA
│   │   ├── preprocessing_models.py    # GroundingDINO + SAM-3
│   │   └── quantization.py            # INT4/INT8 quantification
│   │
│   ├── training/                      # Module entraînement
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Classe Trainer principale
│   │   ├── phase1_projector.py        # Phase 1: alignement
│   │   ├── phase2_lora.py             # Phase 2: LoRA fine-tuning
│   │   ├── callbacks.py               # Callbacks (checkpointing, logging)
│   │   └── utils.py                   # Utilitaires training
│   │
│   ├── evaluation/                    # Module évaluation
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Accuracy, F1, BLEU, ROUGE
│   │   ├── visualizations.py          # Confusion matrix, courbes
│   │   └── evaluator.py               # Classe Evaluator
│   │
│   ├── deployment/                    # Module déploiement
│   │   ├── __init__.py
│   │   ├── api_server.py              # FastAPI cloud
│   │   ├── edge_inference.py          # Inference Jetson/RasPi
│   │   └── mobile_export.py           # Export TFLite
│   │
│   └── utils/                         # Utilitaires généraux
│       ├── __init__.py
│       ├── config.py                  # Chargement configs YAML
│       ├── logging.py                 # Logging structuré
│       └── device.py                  # Gestion GPU/CPU
│
├── scripts/                           # 🚀 Scripts CLI
│   ├── download_data.sh               # Téléchargement datasets
│   ├── preprocess_images.py           # Prétraitement batch
│   ├── generate_multimodal_data.py    # Génération paires
│   ├── train_phase1.py                # Lancement phase 1
│   ├── train_phase2.py                # Lancement phase 2
│   ├── evaluate_model.py              # Évaluation complète
│   └── export_model.py                # Export quantifié
│
├── app/                               # 🖥️ Applications déploiement
│   ├── gradio_app.py                  # Interface Gradio web
│   ├── fastapi_server.py              # API REST production
│   └── mobile_flutter/                # App mobile Flutter
│       └── (structure Flutter)
│
├── tests/                             # 🧪 Tests unitaires
│   ├── test_data/
│   ├── test_models/
│   ├── test_training/
│   └── test_evaluation/
│
├── models/                            # 💾 Modèles sauvegardés (gitignored)
│   ├── checkpoints/                   # Checkpoints entraînement
│   ├── final/                         # Modèle final
│   └── quantized/                     # Modèles quantifiés
│
├── outputs/                           # 📊 Résultats (gitignored)
│   ├── logs/                          # Logs TensorBoard
│   ├── metrics/                       # Métriques JSON/CSV
│   ├── visualizations/                # Graphiques
│   └── reports/                       # Rapports PDF
│
└── docs/                              # 📚 Documentation
    ├── architecture.md                # Architecture système
    ├── training_guide.md              # Guide entraînement
    ├── deployment_guide.md            # Guide déploiement
    └── api_reference.md               # Référence API
```

### 1.2 Fichiers de configuration essentiels

**`requirements.txt`** (principales dépendances):
```txt
# Core ML/DL
torch>=2.0.0
transformers>=4.36.0
peft>=0.7.0
bitsandbytes>=0.41.0
accelerate>=0.25.0

# Vision-Language Models
qwen-vl-utils
pillow>=10.0.0
opencv-python>=4.8.0

# Preprocessing
groundingdino-py
segment-anything>=1.0

# Data
datasets>=2.15.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0

# Deployment
fastapi>=0.104.0
uvicorn>=0.24.0
gradio>=4.0.0

# Utilities
pyyaml>=6.0
tqdm>=4.66.0
wandb>=0.16.0  # Optionnel: tracking experiments
tensorboard>=2.15.0
```

**`.gitignore`**:
```gitignore
# Data
data/raw/*
data/processed/*
data/multimodal/*

# Models
models/checkpoints/*
models/final/*
models/quantized/*

# Outputs
outputs/logs/*
outputs/metrics/*
outputs/visualizations/*

# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
env/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

### 1.3 Configuration Git et branches

```bash
# Créer branches de développement
git checkout -b dev/data-pipeline          # Pipeline données
git checkout -b dev/model-training         # Entraînement
git checkout -b dev/evaluation             # Évaluation
git checkout -b dev/deployment             # Déploiement

# Maintenir main comme branche stable
```

---

## PHASE 2: Collecte et Préparation des Données (Jours 4-10)

### 2.1 Téléchargement des datasets

**Fichier**: `scripts/download_data.sh`

Sources à télécharger:
1. **Kaggle Cassava Leaf Disease** (21,367 images)
   - URL: https://www.kaggle.com/competitions/cassava-leaf-disease-classification
   - Classes: 5 (CMD, CBSD, CBB, CGM, Healthy)

2. **IITA Tanzania Dataset** (15,000 images)
   - Contact IITA ou PlantVillage
   - Classes: 6 (+ sévérité)

3. **Makerere University** (9,103 images)
   - Annotations bounding boxes disponibles

**Total**: ~45,000 images

### 2.2 Analyse exploratoire (EDA)

**Notebook**: `notebooks/01_data_exploration.ipynb`

Analyses à effectuer:
- Distribution des classes (déséquilibre?)
- Statistiques images (résolution, format, qualité)
- Exemples visuels par classe
- Identification de données bruitées
- Stratégie de split stratifié (80% train, 10% val, 10% test)

### 2.3 Pipeline de prétraitement

**Notebook**: `notebooks/02_preprocessing_pipeline.ipynb`
**Code production**: `src/data/preprocessors.py`

#### Étape 1: GroundingDINO (détection feuilles)

```python
# Configuration
prompts = ["cassava leaf", "manioc leaf", "plant leaf"]
box_threshold = 0.35  # Seuil détection
text_threshold = 0.25  # Seuil texte

# Output: Bounding boxes des feuilles
```

#### Étape 2: SAM-3 (segmentation)

```python
# Input: Bounding boxes de GroundingDINO
# Output: Masques pixel-précis des feuilles
# Post-traitement:
#   - Filtrage par surface minimale
#   - Validation dominance canal vert
#   - Stratégie: Recadrage (cropping) recommandée
```

**Métriques qualité prétraitement**:
- Taux de détection réussie (cible: >95%)
- IoU moyen des masques
- Latence par image (cible: <500ms sur RTX 5090)

### 2.4 Génération du corpus multimodal

**Notebook**: `notebooks/03_data_generation.ipynb`
**Code production**: `src/data/multimodal_generator.py`

#### Format LLaVA (paires image-texte)

```json
{
  "image": "cassava_001.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nQuelle maladie affecte cette feuille de manioc ?"
    },
    {
      "from": "gpt",
      "value": "Cette feuille présente les symptômes caractéristiques de la Mosaïque du manioc (CMD). Les signes visibles incluent : 1) Motif de mosaïque jaune-vert sur la surface foliaire, 2) Déformation et réduction de la taille des feuilles, 3) Marbrure irrégulière avec alternance de zones vert clair et vert foncé. Il est recommandé de retirer les plantes infectées et d'utiliser des variétés résistantes."
    }
  ]
}
```

#### Stratégie de génération

1. **Templates de questions** (3-5 par image):
   - Identification: "Quelle maladie affecte cette plante ?"
   - Description: "Décris les symptômes visibles"
   - Recommandations: "Quel traitement recommandes-tu ?"
   - Sévérité: "Évalue la sévérité de l'infection"
   - Prévention: "Comment prévenir cette maladie ?"

2. **Génération via LLM** (GPT-4 ou Claude):
   - Input: Label de classe + image
   - Output: Réponse détaillée en français
   - Révision manuelle d'un échantillon (10%)

3. **Traduction si nécessaire**:
   - Modèle: NLLB-200-distilled-600M
   - Glossaire: CMD→"Mosaïque du manioc", CBSD→"Striure brune", etc.

**Volume cible**: 150,000-225,000 paires (3-5 par image)

### 2.5 Augmentation de données

**Code**: `src/data/augmentation.py`

Techniques:
```python
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ColorJitter(p=0.3),
    A.GaussNoise(p=0.2),
])
```

**Important**: Appliquer APRÈS segmentation pour préserver les feuilles

---

## PHASE 3: Configuration et Chargement du Modèle (Jours 11-14)

### 3.1 Téléchargement Qwen2.5-VL-7B

**Notebook**: `notebooks/04_model_loading.ipynb`

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

# Chargement en FP16 (RTX 5090 32GB → pas de souci)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_name)
```

**Vérifications**:
- VRAM consommée: ~16-18GB (OK pour RTX 5090)
- Inference test sur image exemple
- Latence baseline (~1-2s par image)

### 3.2 Configuration LoRA

**Notebook**: `notebooks/05_lora_setup.ipynb`
**Code**: `src/models/lora_config.py`

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # Rang (comme TLDVLM)
    lora_alpha=32,           # Scaling factor = 2×r
    lora_dropout=0.05,       # Régularisation
    bias="none",
    target_modules=[         # Cibler Q-Former
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    task_type="CAUSAL_LM"
)

# Appliquer LoRA au modèle
model = get_peft_model(model, lora_config)

# Vérification: nombre de paramètres entraînables
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params entraînables: {trainable_params:,}")  # Cible: ~600K
```

**Fichier config**: `configs/model/lora.yaml`

```yaml
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
  bias: none
  task_type: CAUSAL_LM
```

---

## PHASE 4: Entraînement (Jours 15-25)

### 4.1 Phase 1: Alignement du projecteur (1 epoch)

**Notebook**: `notebooks/06_training_phase1.ipynb`
**Script**: `scripts/train_phase1.py`
**Config**: `configs/training/phase1_projector.yaml`

#### Objectif
Aligner les features visuelles avec l'espace du LLM sans modifier le Q-Former.

#### Configuration

```yaml
# configs/training/phase1_projector.yaml
training:
  num_epochs: 1
  batch_size: 4              # Ajuster selon VRAM
  gradient_accumulation: 4   # Batch effectif = 16
  learning_rate: 1e-3        # LR élevé pour alignement rapide

  optimizer:
    type: AdamW
    weight_decay: 0.01
    betas: [0.9, 0.999]

  scheduler:
    type: cosine
    warmup_steps: 100

  mixed_precision: bf16       # bfloat16 pour RTX 5090
  gradient_checkpointing: true

freezing:
  vision_encoder: true        # ✅ Gelé
  q_former: true              # ✅ Gelé
  llm: true                   # ✅ Gelé
  projector: false            # ❌ Entraîné
```

#### Code entraînement

```python
from src.training.phase1_projector import Phase1Trainer

trainer = Phase1Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config="configs/training/phase1_projector.yaml"
)

trainer.train()
```

**Durée estimée**: 3-4 heures sur RTX 5090
**Checkpoint**: `models/checkpoints/phase1_final/`

### 4.2 Phase 2: Fine-tuning LoRA (2-3 epochs)

**Notebook**: `notebooks/07_training_phase2.ipynb`
**Script**: `scripts/train_phase2.py`
**Config**: `configs/training/phase2_lora.yaml`

#### Configuration

```yaml
# configs/training/phase2_lora.yaml
training:
  num_epochs: 3
  batch_size: 2              # Plus petit car LoRA active
  gradient_accumulation: 8   # Batch effectif = 16
  learning_rate: 2e-4        # LR plus faible

  optimizer:
    type: AdamW
    weight_decay: 0.01

  scheduler:
    type: cosine
    warmup_ratio: 0.03
    num_cycles: 0.5

  mixed_precision: bf16
  gradient_checkpointing: true

  # Early stopping
  early_stopping:
    patience: 2
    metric: val_loss
    mode: min

freezing:
  vision_encoder: true        # ✅ Gelé
  q_former: false             # ❌ LoRA appliqué
  llm: false                  # ❌ LoRA appliqué
  projector: false            # ❌ Entraîné

lora:
  enabled: true
  config_path: configs/model/lora.yaml
```

#### Code entraînement

```python
from src.training.phase2_lora import Phase2Trainer

trainer = Phase2Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config="configs/training/phase2_lora.yaml",
    checkpoint_dir="models/checkpoints/phase2/"
)

# Logging avec TensorBoard et WandB
trainer.train(
    log_tensorboard=True,
    log_wandb=True,
    project_name="cassava-vlm"
)
```

**Durée estimée**: 12-18 heures sur RTX 5090 (3 epochs)
**Checkpoints**: Sauvegarde à chaque epoch + best model

### 4.3 Monitoring de l'entraînement

**TensorBoard**:
```bash
tensorboard --logdir outputs/logs/
```

**Métriques à surveiller**:
- Train loss / Val loss (divergence = overfitting)
- Learning rate (cosine decay)
- Gradient norm (stabilité)
- Accuracy par classe (déséquilibre?)
- Perplexity (qualité génération texte)

### 4.4 (Optionnel) Phase 3: Full fine-tuning

**Config**: `configs/training/phase3_full.yaml`

⚠️ **Risque**: Catastrophic forgetting
⚠️ **Utiliser uniquement si** Phase 2 ne donne pas satisfaction

```yaml
training:
  num_epochs: 1
  learning_rate: 1e-6        # Très faible

freezing:
  vision_encoder: true        # ✅ Gelé (toujours)
  q_former: false             # ❌ Entraîné
  llm: false                  # ❌ Entraîné
```

---

## PHASE 5: Évaluation (Jours 26-28)

### 5.1 Métriques de classification

**Notebook**: `notebooks/08_evaluation.ipynb`
**Script**: `scripts/evaluate_model.py`
**Code**: `src/evaluation/metrics.py`

#### Métriques à calculer

```python
from src.evaluation.evaluator import CassavaEvaluator

evaluator = CassavaEvaluator(
    model=model,
    test_dataset=test_dataset,
    device="cuda"
)

results = evaluator.evaluate()

# Métriques
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall: {results['recall']:.4f}")
print(f"F1-Score: {results['f1_score']:.4f}")

# Par classe
for class_name, metrics in results['per_class'].items():
    print(f"{class_name}: F1={metrics['f1']:.4f}")

# Matrice de confusion
evaluator.plot_confusion_matrix(save_path="outputs/visualizations/confusion_matrix.png")

# Courbes ROC
evaluator.plot_roc_curves(save_path="outputs/visualizations/roc_curves.png")
```

**Cibles de performance** (basées sur TLDVLM):
- Accuracy: >95% (cible: 97%)
- F1-score moyen: >0.95
- Pas de classe <90% F1-score

### 5.2 Métriques de génération textuelle

```python
# BLEU, ROUGE, BERTScore pour évaluer qualité des réponses
from src.evaluation.text_metrics import TextMetrics

text_eval = TextMetrics(model, test_dataset)

scores = text_eval.compute_all_metrics()
print(f"BLEU-4: {scores['bleu4']:.4f}")
print(f"ROUGE-L: {scores['rouge_l']:.4f}")
print(f"BERTScore F1: {scores['bertscore_f1']:.4f}")
```

### 5.3 Comparaison avec baselines

**Baselines à implémenter**:
1. **ConvNeXT-tiny** (CNN baseline)
2. **CLIP-LoRA** (VLM alternatif)
3. **EfficientNet-B4** (CNN état de l'art cassava)

**Tableau comparatif**:

| Modèle | Accuracy | F1-Score | Params entraînables | VRAM | Latence |
|--------|----------|----------|---------------------|------|---------|
| ConvNeXT-tiny | - | - | - | - | - |
| CLIP-LoRA | - | - | - | - | - |
| EfficientNet-B4 | - | - | - | - | - |
| **CassavaVLM (Ours)** | **?** | **?** | **0.6M** | **18GB** | **~2s** |

### 5.4 Analyse des erreurs

```python
# Identifier les cas d'erreurs
errors = evaluator.get_misclassifications()

# Visualiser exemples d'erreurs
evaluator.visualize_errors(
    num_samples=20,
    save_dir="outputs/visualizations/errors/"
)

# Analyser patterns d'erreurs
# - Confusion entre classes visuellement similaires?
# - Problèmes de segmentation?
# - Images de mauvaise qualité?
```

---

## PHASE 6: Déploiement (Jours 29-35)

### 6.1 API Cloud (FastAPI)

**Fichier**: `app/fastapi_server.py`

```python
from fastapi import FastAPI, File, UploadFile
from src.models.qwen_vlm import CassavaVLM
from src.data.preprocessors import PreprocessingPipeline

app = FastAPI(title="CassavaVLM API")

# Charger modèle au démarrage
model = CassavaVLM.from_pretrained("models/final/cassava_vlm_best")
preprocessor = PreprocessingPipeline()

@app.post("/predict")
async def predict_disease(image: UploadFile = File(...)):
    # 1. Prétraiter image
    processed_image = await preprocessor.process(image)

    # 2. Inference
    result = model.predict(processed_image)

    return {
        "disease": result["predicted_class"],
        "confidence": result["confidence"],
        "description": result["description"],
        "recommendations": result["recommendations"]
    }

@app.post("/ask")
async def interactive_qa(image: UploadFile, question: str):
    processed_image = await preprocessor.process(image)
    answer = model.answer_question(processed_image, question)
    return {"answer": answer}
```

**Déploiement**:
```bash
uvicorn app.fastapi_server:app --host 0.0.0.0 --port 8000
```

### 6.2 Interface Gradio

**Fichier**: `app/gradio_app.py`

```python
import gradio as gr
from src.models.qwen_vlm import CassavaVLM

model = CassavaVLM.from_pretrained("models/final/cassava_vlm_best")

def diagnose(image, question=None):
    if question:
        # Mode interactif
        answer = model.answer_question(image, question)
        return answer
    else:
        # Mode diagnostic simple
        result = model.predict(image)
        return f"""
        **Maladie détectée**: {result['disease']}
        **Confiance**: {result['confidence']:.2%}

        **Description**: {result['description']}

        **Recommandations**: {result['recommendations']}
        """

iface = gr.Interface(
    fn=diagnose,
    inputs=[
        gr.Image(type="pil", label="Photo de la feuille"),
        gr.Textbox(label="Question (optionnel)", placeholder="Ex: Comment traiter cette maladie?")
    ],
    outputs=gr.Markdown(),
    title="🌿 CassavaDoc - Diagnostic des maladies du manioc",
    description="Système de diagnostic interactif basé sur Vision-Language Model"
)

iface.launch(share=True)
```

### 6.3 Quantification pour edge/mobile

**Script**: `scripts/export_model.py`

```python
from src.models.quantization import quantize_model

# Export GGUF INT4 pour Jetson Orin
quantize_model(
    model_path="models/final/cassava_vlm_best",
    output_path="models/quantized/cassava_vlm_int4.gguf",
    quantization_type="int4",
    target="jetson"
)

# Export TFLite INT8 pour mobile
quantize_model(
    model_path="models/final/cassava_vlm_best",
    output_path="models/quantized/cassava_vlm_int8.tflite",
    quantization_type="int8",
    target="mobile"
)
```

---

## PHASE 7: Documentation et Tests (Jours 36-40)

### 7.1 Documentation technique

**Fichiers à créer**:

1. **`docs/architecture.md`**
   - Schéma architecture globale
   - Pipeline de données
   - Architecture modèle
   - Déploiement multi-plateforme

2. **`docs/training_guide.md`**
   - Guide reproduction entraînement
   - Hyperparamètres détaillés
   - Troubleshooting

3. **`docs/deployment_guide.md`**
   - Déploiement cloud
   - Déploiement edge (Jetson)
   - App mobile Flutter

4. **`README.md`** complet
   - Présentation projet
   - Installation
   - Quick start
   - Résultats
   - Citation

### 7.2 Tests unitaires

**Structure**: `tests/`

```python
# tests/test_preprocessing.py
def test_groundingdino_detection():
    # Test détection feuilles
    pass

def test_sam3_segmentation():
    # Test segmentation
    pass

# tests/test_model.py
def test_model_loading():
    # Test chargement Qwen2.5-VL
    pass

def test_lora_config():
    # Test configuration LoRA
    pass

# tests/test_training.py
def test_training_step():
    # Test une étape d'entraînement
    pass

# tests/test_evaluation.py
def test_metrics_computation():
    # Test calcul métriques
    pass
```

**Exécution**:
```bash
pytest tests/ -v --cov=src/
```

---

## FICHIERS CRITIQUES À CRÉER (Ordre de priorité)

### Priorité 1: Infrastructure (Jours 1-3)
1. ✅ `requirements.txt` - Dépendances
2. ✅ `.gitignore` - Exclusions Git
3. ✅ `configs/model/qwen2_5_vl_7b.yaml` - Config modèle
4. ✅ `configs/model/lora.yaml` - Config LoRA
5. ✅ `configs/training/phase1_projector.yaml` - Config phase 1
6. ✅ `configs/training/phase2_lora.yaml` - Config phase 2

### Priorité 2: Données (Jours 4-10)
7. ✅ `scripts/download_data.sh` - Téléchargement datasets
8. ✅ `notebooks/01_data_exploration.ipynb` - EDA
9. ✅ `notebooks/02_preprocessing_pipeline.ipynb` - Prétraitement
10. ✅ `src/data/preprocessors.py` - Code prétraitement
11. ✅ `notebooks/03_data_generation.ipynb` - Génération multimodal
12. ✅ `src/data/multimodal_generator.py` - Code génération

### Priorité 3: Modèle (Jours 11-14)
13. ✅ `notebooks/04_model_loading.ipynb` - Chargement Qwen2.5-VL
14. ✅ `notebooks/05_lora_setup.ipynb` - Setup LoRA
15. ✅ `src/models/qwen_vlm.py` - Wrapper modèle
16. ✅ `src/models/lora_config.py` - Config LoRA

### Priorité 4: Entraînement (Jours 15-25)
17. ✅ `notebooks/06_training_phase1.ipynb` - Entraînement phase 1
18. ✅ `notebooks/07_training_phase2.ipynb` - Entraînement phase 2
19. ✅ `src/training/trainer.py` - Classe Trainer
20. ✅ `src/training/phase1_projector.py` - Code phase 1
21. ✅ `src/training/phase2_lora.py` - Code phase 2
22. ✅ `scripts/train_phase1.py` - Script CLI phase 1
23. ✅ `scripts/train_phase2.py` - Script CLI phase 2

### Priorité 5: Évaluation (Jours 26-28)
24. ✅ `notebooks/08_evaluation.ipynb` - Évaluation
25. ✅ `src/evaluation/evaluator.py` - Code évaluation
26. ✅ `src/evaluation/metrics.py` - Métriques
27. ✅ `scripts/evaluate_model.py` - Script CLI évaluation

### Priorité 6: Déploiement (Jours 29-35)
28. ✅ `app/fastapi_server.py` - API cloud
29. ✅ `app/gradio_app.py` - Interface Gradio
30. ✅ `src/deployment/api_server.py` - Code API
31. ✅ `scripts/export_model.py` - Export quantifié

---

## TIMELINE DÉTAILLÉE (40 jours)

| Phase | Jours | Tâches | Livrables |
|-------|-------|--------|-----------|
| **Phase 1: Infrastructure** | 1-3 | Setup projet, configs, dépendances | Structure projet complète |
| **Phase 2: Données** | 4-10 | Téléchargement, EDA, prétraitement, génération multimodal | 45K images + 150K paires |
| **Phase 3: Modèle** | 11-14 | Chargement Qwen2.5-VL, setup LoRA | Modèle prêt à entraîner |
| **Phase 4: Entraînement** | 15-25 | Phase 1 (1 epoch), Phase 2 (3 epochs) | Modèle fine-tuné |
| **Phase 5: Évaluation** | 26-28 | Métriques, comparaisons, analyse erreurs | Rapport performance |
| **Phase 6: Déploiement** | 29-35 | API, Gradio, quantification | Apps déployées |
| **Phase 7: Documentation** | 36-40 | Docs, tests, finalisation | Projet complet |

---

## MÉTRIQUES DE SUCCÈS

### Objectifs techniques
- ✅ **Accuracy**: >95% (cible: 97% comme TLDVLM)
- ✅ **F1-score moyen**: >0.95
- ✅ **Latence cloud**: <2s par diagnostic
- ✅ **VRAM training**: <32GB (OK RTX 5090)
- ✅ **Paramètres LoRA**: ~0.6M (<2% du modèle)

### Objectifs fonctionnels
- ✅ Diagnostic interactif en français
- ✅ Réponses contextuelles cohérentes
- ✅ Interface utilisateur intuitive
- ✅ Déploiement multi-plateforme

---

## RISQUES ET MITIGATION

| Risque | Impact | Probabilité | Mitigation |
|--------|--------|-------------|------------|
| Overfitting (dataset limité) | Élevé | Moyenne | Augmentation données, early stopping, LoRA faible rang |
| Déséquilibre classes | Moyen | Élevée | Weighted loss, oversampling minoritaires |
| Latence élevée inference | Moyen | Faible | Quantification INT4, optimisation pipeline |
| Qualité réponses françaises | Moyen | Moyenne | Validation humaine, fine-tuning corpus bilingue |
| VRAM insuffisante | Élevé | Faible | Gradient checkpointing, batch size adaptatif |

---

## PROCHAINES ÉTAPES IMMÉDIATES

### Jour 1 (Aujourd'hui)
1. ✅ Créer structure de répertoires
2. ✅ Créer `requirements.txt`
3. ✅ Créer `.gitignore`
4. ✅ Initialiser repo Git avec branches
5. ✅ Créer premiers fichiers de config YAML

### Jour 2
6. ✅ Créer `scripts/download_data.sh`
7. ✅ Télécharger Kaggle Cassava dataset
8. ✅ Créer `notebooks/01_data_exploration.ipynb`
9. ✅ Analyser distribution des données

### Jour 3
10. ✅ Créer `notebooks/02_preprocessing_pipeline.ipynb`
11. ✅ Tester GroundingDINO + SAM-3 sur échantillon
12. ✅ Valider pipeline prétraitement

**Commencer l'implémentation dès validation de ce plan!**
