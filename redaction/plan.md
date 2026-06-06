# PLAN DU MÉMOIRE D'INGÉNIEUR

## Titre
**CassavaVLM : Premier Vision-Language Model multilingue interactif pour le diagnostic des maladies du manioc**

---

## PAGES LIMINAIRES

- Page de garde
- Dédicaces
- Remerciements
- Résumé (Français)
- Abstract (Anglais)
- Table des matières
- Liste des figures
- Liste des tableaux
- Liste des abréviations et sigles

---

## INTRODUCTION GÉNÉRALE

1. Contexte et justification
2. Problématique
3. Hypothèses de recherche
4. Objectifs
   - Objectif général
   - Objectifs spécifiques
5. Méthodologie résumée
6. Structure du mémoire

---

## CHAPITRE 1 : SYNTHÈSE BIBLIOGRAPHIQUE

### 1.1 Le manioc : importance et défis phytosanitaires

#### 1.1.1 Importance économique et alimentaire du manioc
- Rôle du manioc en Afrique subsaharienne (500 millions de consommateurs)
- Place du manioc dans l'agriculture béninoise
- Chaîne de valeur et enjeux socio-économiques
- Impact des changements climatiques sur la culture

#### 1.1.2 Principales maladies du manioc
- Mosaïque du manioc (CMD) : symptômes, transmission, impact
- Striure brune du manioc (CBSD) : distribution et sévérité
- Bactériose du manioc (CBB) : caractéristiques et propagation
- Acarien vert du manioc (CGM) : dégâts et reconnaissance
- Feuilles saines : critères de distinction

#### 1.1.3 Méthodes traditionnelles de diagnostic
- Inspection visuelle par experts phytopathologistes
- Limites des approches conventionnelles (subjectivité, disponibilité)
- Besoins des agriculteurs africains : interactivité, multilingue, offline

### 1.2 Deep learning pour la détection des maladies du manioc

#### 1.2.1 Évolution des approches de vision par ordinateur
- Méthodes classiques (extraction de caractéristiques manuelles)
- Réseaux de neurones convolutifs (CNN) : AlexNet à EfficientNet
- Vision Transformers (ViT) : paradigme attention-based
- Architectures hybrides modernes (ConvNeXt, Swin Transformer)

#### 1.2.2 État de l'art sur la détection des maladies du manioc
- Travaux fondateurs : Ramcharan et al. (2017), Mwebaze et al. (2019)
- Performances CNN actuelles : MiXceptionLeaf (98.8%), CDDNet (98.95%)
- Vision Transformers pour cassava : FormerLeaf (91.43%), ED-Swin (94.32%)
- Applications déployées : PlantVillage Nuru (50,000+ agriculteurs)

#### 1.2.3 Datasets disponibles
- Kaggle Cassava Leaf Disease (21,397 images, 5 classes)
- Makerere University Cassava Image Dataset
- Datasets terrain récents (Tanzania, UAV imaging)

#### 1.2.4 Limites des approches CNN/ViT actuelles
- Sortie limitée aux labels de classification
- Absence d'interactivité et d'explications textuelles
- Pas de support multilingue natif
- Conseils agronomiques statiques pré-écrits

### 1.3 Vision-Language Models : une nouvelle frontière

#### 1.3.1 Fondements des modèles vision-langage
- Principe de l'apprentissage multimodal (vision + texte)
- CLIP : apprentissage contrastif image-texte (Radford et al., 2021)
- BLIP et BLIP-2 : architecture Q-Former (Li et al., 2023)
- Mécanismes d'attention croisée vision-texte

#### 1.3.2 Visual Instruction Tuning
- Paradigme LLaVA (Liu et al., NeurIPS 2023 Oral)
- Pipeline en deux stages : feature alignment → instruction tuning
- Génération de données d'instruction synthétiques
- Capacités émergentes : VQA, raisonnement, dialogue multi-tour

#### 1.3.3 Qwen2.5-VL : architecture et capacités
- Structure : ViT modifié + M-RoPE + Qwen2.5 LLM
- Native Dynamic Resolution et Window Attention
- Support multilingue natif (29+ langues dont français)
- Performances état de l'art sur benchmarks VQA

#### 1.3.4 Applications des VLM en agriculture
- AgroBench : premier benchmark VLM agricole (ICCV 2025)
- Agri-LLaVA : fine-tuning sur 391K échantillons agricoles
- TLDVLM : diagnostic de la tomate (97.27% accuracy)
- PlantVillageVQA : dataset VQA agricole (193K paires)
- Gap identifié : aucun VLM spécifique au manioc

#### 1.3.5 ITIMCA : seul travail multimodal sur le manioc
- Approche CLIP pour classification contrastive
- Performances : 78% accuracy, 88.48% précision
- Limitations : pas de VQA conversationnel, anglais uniquement

### 1.4 Techniques d'adaptation efficace des grands modèles

#### 1.4.1 Low-Rank Adaptation (LoRA)
- Principe de la décomposition en rang faible (Hu et al., 2022)
- Avantages : réduction mémoire 75%, préservation des connaissances
- Paramètres clés : rang r, facteur alpha, modules cibles
- Application aux VLMs : q_proj, k_proj, v_proj, o_proj

#### 1.4.2 QLoRA et quantification
- Quantification NormalFloat 4-bit (Dettmers et al., 2023)
- Double quantification pour réduction mémoire supplémentaire
- Compromis performance/efficacité
- Entraînement sur GPU consumer (~7GB VRAM)

#### 1.4.3 Stratégies de fine-tuning multi-stage
- Curriculum learning pour VLMs (Curr-ReFT)
- Approche LLaVA-Med : concept alignment intermédiaire
- Prévention de l'oubli catastrophique
- Transfer learning entre stages

### 1.5 Déploiement en contexte africain

#### 1.5.1 Contraintes techniques du terrain
- Connectivité limitée : nécessité offline-first
- Smartphones bas de gamme : 4GB RAM, stockage limité
- Coût énergétique et batteries
- Leçons de PlantVillage Nuru

#### 1.5.2 Quantification pour déploiement mobile
- Formats : GGUF, AWQ, GPTQ
- Comparaison INT4 vs INT8
- Impact sur latence et qualité
- Frameworks : llama.cpp, TensorFlow Lite

#### 1.5.3 Architectures de déploiement
- Mode cloud : API vLLM pour modèle complet
- Mode edge : Jetson pour centres de vulgarisation
- Mode mobile : modèle quantifié sur smartphone
- Synchronisation hybride cloud-edge-mobile

---

## CHAPITRE 2 : MATÉRIELS ET MÉTHODES

### 2.1 Environnement matériel et logiciel

#### 2.1.1 Configuration matérielle
- Station d'entraînement (GPU, CPU, RAM)
- Dispositifs de déploiement cibles
- Ressources cloud utilisées

#### 2.1.2 Environnement logiciel
- Système d'exploitation (Ubuntu 24.04)
- Frameworks deep learning (PyTorch, Transformers, PEFT)
- Bibliothèque Unsloth pour optimisation
- Outils de développement et versioning

### 2.2 Données

#### 2.2.1 Sources de données
- Dataset Kaggle Cassava Leaf Disease (21,367 images)
- Distribution des 5 classes : CMD, CBSD, CBB, CGM, Healthy
- Analyse du déséquilibre des classes

#### 2.2.2 Préparation des données pour Stage 1 (Classification)
- Partitionnement train/validation/test
- Stratégie d'oversampling (CBSD 3x, Healthy 2x)
- Augmentation de données (Albumentations)
- Format de prompt multiple-choice (A/B/C/D/E)

#### 2.2.3 Construction du dataset d'instructions pour Stage 2 (VQA)
- Génération synthétique via LLM
- Types d'instructions : diagnostic, description, raisonnement, conseils
- Traduction et adaptation au français agricole
- Format conversationnel multi-tour
- Validation par expertise agronomique

### 2.3 Architecture du modèle CassavaVLM

#### 2.3.1 Choix du modèle de base : Qwen2.5-VL-7B
- Justification : multilingue, performance, efficacité
- Comparaison avec alternatives (LLaVA, InternVL, BLIP-2)
- Version quantifiée : unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit

#### 2.3.2 Configuration LoRA
- Modules cibles : q_proj, v_proj
- Hyperparamètres : r=8, α=8, dropout=0
- Justification du ratio conservatif 1:1

#### 2.3.3 Architecture globale du pipeline
- Encodeur visuel (ViT) : gelé pendant fine-tuning
- Projecteur multimodal
- Décodeur LLM avec adaptateurs LoRA

### 2.4 Protocole d'entraînement multi-stage

#### 2.4.1 Stage 1 : Classification des maladies
- Objectif : alignement vision-classification
- Configuration d'entraînement (epochs, batch size, learning rate)
- Scheduler et optimiseur (AdamW, warmup)
- Techniques d'optimisation (mixed precision bf16, gradient checkpointing)

#### 2.4.2 Stage 2 : VQA interactif et conseils agronomiques
- Chargement des poids Stage 1
- Fine-tuning sur dataset d'instructions
- Adaptation pour génération conversationnelle
- Gestion du contexte multi-tour

#### 2.4.3 Stratégies d'optimisation mémoire
- Gradient accumulation (effective batch size)
- Mixed precision training
- Gradient checkpointing

### 2.5 Métriques d'évaluation

#### 2.5.1 Métriques de classification (Stage 1)
- Accuracy globale et par classe
- Précision, Rappel, F1-score
- Matrice de confusion
- Analyse des classes difficiles

#### 2.5.2 Métriques de génération textuelle (Stage 2)
- Exact Match pour diagnostic
- BLEU et ROUGE-L pour qualité textuelle
- BERTScore pour similarité sémantique
- Évaluation humaine : pertinence, exactitude, utilité

#### 2.5.3 Métriques de déploiement
- Latence (temps d'inférence)
- Consommation mémoire GPU/RAM
- Tokens par seconde
- Taille des modèles quantifiés

### 2.6 Protocole de quantification et déploiement

#### 2.6.1 Méthodes de quantification testées
- AWQ INT4 pour serveur GPU
- GGUF Q4_K_M pour mobile/CPU
- Impact sur accuracy et latence

#### 2.6.2 Architecture de déploiement proposée
- API cloud (FastAPI + vLLM)
- Application mobile offline-first
- Synchronisation différée

---

## CHAPITRE 3 : RÉSULTATS ET DISCUSSIONS

### 3.1 Résultats Stage 1 : Classification des maladies

#### 3.1.1 Performances globales
- Accuracy atteinte : 80.52%
- Évolution de la loss pendant l'entraînement
- Courbes d'apprentissage

#### 3.1.2 Analyse par classe
- F1-score par maladie
- Amélioration critique CBSD : 2.9% → 84% (+81 points)
- Performances CMD, CGM, CBB, Healthy
- Matrice de confusion détaillée

#### 3.1.3 Comparaison avec les baselines
- Versus baseline Gemma 3n (2B)
- Versus CNNs SOTA (EfficientNet, ConvNeXt)
- Versus approche ITIMCA (78%)
- Analyse des gains et écarts

#### 3.1.4 Discussion Stage 1
- Impact de l'oversampling sur CBSD
- Rôle de la configuration LoRA conservatrice
- Effet de l'augmentation de données
- Limitations et axes d'amélioration

### 3.2 Résultats Stage 2 : VQA interactif

#### 3.2.1 Performances de diagnostic conversationnel
- Exact Match sur questions de classification
- Qualité des explications générées
- Cohérence des dialogues multi-tour

#### 3.2.2 Qualité des recommandations agronomiques
- Évaluation BLEU/ROUGE des conseils
- BERTScore pour pertinence sémantique
- Comparaison avec conseils experts IITA/FAO

#### 3.2.3 Évaluation humaine
- Protocole d'évaluation par experts
- Scores de pertinence et utilité
- Feedback qualitatif
- Cas d'erreurs et hallucinations

#### 3.2.4 Discussion Stage 2
- Apport du fine-tuning multi-stage
- Qualité du français généré
- Limites de la génération
- Comparaison avec conseils statiques (Nuru)

### 3.3 Évaluation comparative globale

#### 3.3.1 CassavaVLM vs approches existantes
- Tableau comparatif : CNN, ViT, VLM zero-shot, CassavaVLM
- Avantages de l'interactivité
- Valeur ajoutée du multilingue

#### 3.3.2 Ablation study
- Impact de chaque stage
- Effet des hyperparamètres LoRA
- Contribution de l'oversampling
- Rôle de l'augmentation

#### 3.3.3 Analyse qualitative
- Exemples de dialogues réussis
- Cas d'échecs et raisons
- Visualisation des réponses

### 3.4 Résultats de quantification et déploiement

#### 3.4.1 Impact de la quantification
- Comparaison FP16 vs INT8 vs INT4
- Rétention de performance par méthode
- Trade-off accuracy/latence/mémoire

#### 3.4.2 Performances par plateforme
- Latence sur GPU serveur
- Latence sur smartphone représentatif
- Consommation mémoire

#### 3.4.3 Faisabilité du déploiement offline
- Tests en conditions simulées terrain
- Autonomie et consommation batterie
- Expérience utilisateur

### 3.5 Discussion générale

#### 3.5.1 Validation des hypothèses de recherche
- H1 : VLM peut atteindre performances comparables aux CNNs
- H2 : Fine-tuning multi-stage améliore les capacités VQA
- H3 : Déploiement offline est faisable

#### 3.5.2 Contributions principales
- Premier VLM dédié au diagnostic du manioc
- Approche multi-stage classification → VQA
- Support multilingue français génératif
- Pipeline de déploiement adapté au contexte africain

#### 3.5.3 Limites de l'étude
- Taille et diversité du dataset
- Absence de validation terrain réelle
- Généralisation géographique
- Couverture des variétés de manioc

#### 3.5.4 Perspectives et travaux futurs
- Amélioration de l'accuracy classification
- Extension aux langues locales (Fon, Yoruba, Swahili)
- Évaluation de la sévérité des maladies
- Intégration système d'alerte épidémiologique
- Partenariats institutionnels (INRAB, IITA)
- Validation avec agriculteurs béninois

---

## CONCLUSION GÉNÉRALE

1. Synthèse des travaux réalisés
2. Contributions principales
3. Limites et perspectives
4. Mot de fin

---

## RÉFÉRENCES BIBLIOGRAPHIQUES

---

## ANNEXES

- Annexe A : Extraits de code source (configuration LoRA, training loop)
- Annexe B : Configuration détaillée des hyperparamètres
- Annexe C : Exemples complets de dialogues avec le modèle
- Annexe D : Dataset d'instructions VQA (échantillon)
- Annexe E : Protocole d'évaluation humaine
- Annexe F : Résultats détaillés par classe et par fold

---

## TABLE DES MATIÈRES DÉTAILLÉE (estimations de pages)

| Section | Pages estimées |
|---------|----------------|
| Pages liminaires | 10-12 |
| Introduction générale | 4-5 |
| Chapitre 1 : Synthèse bibliographique | 30-35 |
| Chapitre 2 : Matériels et méthodes | 20-25 |
| Chapitre 3 : Résultats et discussions | 35-45 |
| Conclusion générale | 3-4 |
| Références bibliographiques | 6-8 |
| Annexes | 15-20 |
| **TOTAL** | **125-155 pages** |