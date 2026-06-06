# Cassava-VLM — Fine-tuning a Vision-Language Model for Cassava Leaf-Disease Diagnosis

Fine-tuning of **Qwen2.5-VL-7B** with **4-bit QLoRA** to classify and explain five cassava leaf conditions from a single photo, built toward low-resource, offline on-device diagnosis for smallholder farmers.

## Overview

Cassava is a staple crop for hundreds of millions of people, yet field diagnosis of its diseases is usually limited to CNN classifiers that output a label with no explanation. This project fine-tunes an open multimodal LLM so it can both **identify** the disease and **describe** it in natural language, while staying small enough (4-bit quantized, ~6–8 GB) to be a realistic candidate for offline deployment on constrained hardware. The model is trained on a merged, deduplicated cassava image corpus and reaches **~80.5% accuracy** on a held-out 5-class test set.

> Status: research prototype / engineering thesis work. Training, evaluation, and quantized inference are demonstrated in this repo; on-device mobile deployment is a stated design goal, not yet shown here (see Limitations).

## Technical approach

This is the core of the project.

### Model and quantization
- **Base model:** `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit` (Alibaba's Qwen2.5-VL-7B, Apache-2.0), loaded in **4-bit (NF4 via bitsandbytes)** through Unsloth's `FastVisionModel`.
- **Parameter-efficient fine-tuning:** QLoRA — frozen 4-bit base + trainable LoRA adapters, with gradient checkpointing. Only a small fraction of parameters are trained.

### Two-stage fine-tuning (my pipeline)
| Stage | Goal | LoRA | Key hyperparameters |
|-------|------|------|---------------------|
| **Stage 1** — disease foundation | Single-label disease recognition | r=8, α=8, dropout 0, targets `q_proj,v_proj` | lr 2e-5, 3 epochs, batch 2 × grad-accum 8 (eff. 16), cosine + 0.1 warmup, wd 0.01, `adamw_8bit`, bf16 |
| **Stage 2** — unified multi-task (V5) | Classification **+** VQA-style conversations | r=32, α=32, dropout 0.05, targets `q,k,v,o,gate,up,down` | lr 1e-5, 2 epochs, same batch geometry, early stopping (patience 3) |

- Images resized to **512×512**; **Albumentations** augmentation in Stage 1 (random resized crop, flips, rotation, HSV/brightness jitter, CLAHE, noise, coarse dropout).
- Notebooks: [04_train_stage1_cassava_qwen25vl_7b.ipynb](new_implementation/notebooks/04_train_stage1_cassava_qwen25vl_7b.ipynb), [10_train_stage2_v5_unified_multitask_final.ipynb](new_implementation/notebooks/10_train_stage2_v5_unified_multitask_final.ipynb).

### Data (my pipeline)
- **Merged corpus:** the Kaggle *Cassava Leaf Disease* set (21,397 images) plus additional cassava/CBSD image folders, MD5-deduplicated into ~33k images across five classes: **CBB, CBSD, CGM, CMD, Healthy** ([00_prepare_cassava_data.ipynb](new_implementation/notebooks/00_prepare_cassava_data.ipynb)).
- **Class imbalance handling:** minority-class oversampling (notably CBSD, the hardest class).
- **Instruction data:** Stage-2 builds a VQA/conversation dataset on top of the labeled images so the model learns to answer free-form questions, not just emit a class.

### Inference / serving
- Greedy decoding behind a constrained multiple-choice prompt; predictions parsed by a weighted regex extractor that normalizes class names. Runs in the same 4-bit configuration used for training (Unsloth `for_inference`).

## Results / status

Evaluated on a held-out cassava test set (3,619 images) with the fine-tuned Qwen2.5-VL-7B:

| Class | Full name | Accuracy |
|-------|-----------|----------|
| CBB | Cassava Bacterial Blight | 55.1% |
| CBSD | Cassava Brown Streak Disease | 84.0% |
| CGM | Cassava Green Mottle | 56.5% |
| CMD | Cassava Mosaic Disease | 86.0% |
| Healthy | Healthy leaf | 75.4% |
| **Global** | — | **80.5%** (2,915/3,619) |

- Final validation loss ≈ 0.07.
- **Evaluation honesty note:** an initial run reported 45.85% accuracy due to an answer-parsing bug (the model's free-text answers weren't always mapped to a class). After fixing the extractor, corrected accuracy is ~80% — see [evaluation_report_final_corrected.txt](new_implementation/evaluation/cassava_vlm_v5_final/reports/evaluation_report_final_corrected.txt).
- Metrics, confusion matrices, and learning curves are reproduced under [new_implementation/results/](new_implementation/results/) and [new_implementation/figures_memoire/](new_implementation/figures_memoire/).

This is a work-in-progress prototype; numbers above are what the repo actually demonstrates.

## Tech stack

- **ML:** PyTorch, Hugging Face Transformers, Unsloth (`FastVisionModel`), PEFT/LoRA, bitsandbytes (4-bit), TRL/`Trainer`, Datasets
- **Vision/data:** Albumentations, Pillow, OpenCV, pandas/NumPy
- **Eval/viz:** scikit-learn, matplotlib, seaborn; Weights & Biases for run tracking
- **Third-party model:** Qwen2.5-VL-7B-Instruct (Apache-2.0); **datasets:** Kaggle Cassava Leaf Disease + additional cassava image sets

## Repository structure

```
cassava_vlm/
├── new_implementation/        # Main, current work (Qwen2.5-VL QLoRA)
│   ├── notebooks/             # Data prep, Stage 1 / Stage 2 training, evaluation
│   ├── data_preparation/      # Dataset build, VQA formatting, replay-buffer scripts
│   ├── results/               # Metrics, confusion matrices, learning curves
│   ├── figures_memoire/       # Final thesis figures and metric JSON/CSV
│   ├── evaluation/            # Corrected evaluation reports
│   └── requirements.txt
├── notebooks/                 # Earlier exploration (zero-shot, EDA, preprocessing)
├── outputs/                   # Artifacts from the exploration phase
├── data/                      # Dataset placeholder (data itself is git-ignored)
├── guide_projet.md            # Literature review / design rationale
├── IMPLEMENTATION_PLAN.md     # Original project plan
└── requirements.txt
```

Note: `new_implementation/README.md` describes an earlier, broader multilingual plan that was not the path ultimately taken; this root README reflects what was actually built.

## Setup & usage

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r new_implementation/requirements.txt
# Unsloth (used for 4-bit Qwen2.5-VL):
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 2. Data — Kaggle Cassava Leaf Disease (needs ~/.kaggle/kaggle.json)
kaggle competitions download -c cassava-leaf-disease-classification
# unzip into data/raw/ (train.csv + train_images/), then build the merged dataset:
jupyter notebook new_implementation/notebooks/00_prepare_cassava_data.ipynb

# 3. Train
jupyter notebook new_implementation/notebooks/04_train_stage1_cassava_qwen25vl_7b.ipynb
jupyter notebook new_implementation/notebooks/10_train_stage2_v5_unified_multitask_final.ipynb

# 4. Evaluate
jupyter notebook new_implementation/notebooks/11_evaluate_v5_checkpoint_final.ipynb
```

A CUDA GPU is required; 4-bit loading keeps the 7B model within roughly 8–12 GB of VRAM for inference. Paths inside the notebooks are absolute (`/home/.../cassava/...`) and should be adjusted to your machine.

## Limitations / roadmap

- **No on-device deployment yet.** 4-bit quantization is in place, but mobile/edge export (e.g. GGUF) and an offline app have not been built — the headline use case is still a goal.
- **Notebook-driven.** Training/eval live in notebooks with hard-coded absolute paths; there is no packaged CLI/library or pinned environment.
- **Weak minority classes.** CBB and CGM (~55–57%) lag behind CMD/CBSD; more balanced data and targeted augmentation are the next lever.
- **Reproducibility gaps.** The merged dataset and checkpoints are git-ignored; only generation scripts are versioned, so exact numbers depend on regenerating the corpus.
- **Single domain.** Cassava-only, five classes; multilingual and multi-crop extensions remain future work.
