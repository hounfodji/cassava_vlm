# PLAN DU MÉMOIRE D'INGÉNIEUR

## Titre
**Diagnostic interactif des maladies du manioc par Vision-Language Model : Une approche multimodale avec support français et déploiement hybride**

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

## CHAPITRE 1 : REVUE DE LITTÉRATURE

### 1.1 Le manioc : importance et défis phytosanitaires

#### 1.1.1 Importance économique et alimentaire du manioc
- Rôle du manioc en Afrique subsaharienne
- Place du manioc dans l'agriculture béninoise
- Chaîne de valeur et enjeux socio-économiques

#### 1.1.2 Principales maladies du manioc
- Mosaïque du manioc (CMD)
- Striure brune du manioc (CBSD)
- Bactériose du manioc (CBB)
- Acarien vert du manioc (CGM)
- Autres maladies et ravageurs

#### 1.1.3 Méthodes traditionnelles de diagnostic
- Inspection visuelle par experts
- Limites des approches conventionnelles
- Besoins en solutions automatisées

### 1.2 Intelligence artificielle pour la détection des maladies végétales

#### 1.2.1 Évolution des approches de vision par ordinateur
- Méthodes classiques (extraction de caractéristiques)
- Réseaux de neurones convolutifs (CNN)
- Architectures modernes (ResNet, EfficientNet, ConvNeXt)

#### 1.2.2 État de l'art sur la détection des maladies du manioc
- Travaux existants et performances atteintes
- Datasets disponibles (Kaggle, Makerere, IITA)
- Limites des approches CNN actuelles

#### 1.2.3 Lacunes identifiées dans la littérature
- Absence d'interactivité et d'explications
- Manque de support multilingue
- Difficultés de déploiement en contexte africain

### 1.3 Vision-Language Models (VLM) : une nouvelle frontière

#### 1.3.1 Fondements des modèles vision-langage
- Principe de l'apprentissage multimodal
- Architectures fondatrices (CLIP, BLIP, Flamingo)
- Mécanismes d'attention croisée vision-texte

#### 1.3.2 BLIP-2 et l'architecture Q-Former
- Structure de BLIP-2 (encodeur visuel, Q-Former, LLM)
- Avantages du Q-Former pour l'alignement multimodal
- Capacités de Visual Question Answering

#### 1.3.3 Qwen2.5-VL et alternatives multilingues
- Architecture et performances de Qwen2.5-VL
- Support natif du français
- Comparaison avec autres VLMs (InternVL, LLaVA)

#### 1.3.4 Applications des VLM en agriculture
- Travaux récents (TLDVLM sur tomate, PlantCareNet)
- Potentiel inexploité pour le manioc
- Opportunité de contribution originale

### 1.4 Techniques d'adaptation efficace des grands modèles

#### 1.4.1 Low-Rank Adaptation (LoRA)
- Principe de la décomposition en rang faible
- Avantages en termes de mémoire et calcul
- Paramètres clés (rang, alpha, modules cibles)

#### 1.4.2 Autres méthodes PEFT
- QLoRA et quantification
- Adapters et Prefix-tuning
- Comparaison des approches

### 1.5 Segmentation et prétraitement d'images

#### 1.5.1 Modèles de détection zero-shot
- GroundingDINO : détection guidée par texte
- Applications en agriculture

#### 1.5.2 Segment Anything Model (SAM-3)
- Architecture et capacités
- Segmentation promptée par boîtes englobantes

#### 1.5.3 Impact du prétraitement sur les performances
- Études comparatives
- Pipeline optimal pour les feuilles

### 1.6 Génération de recommandations par RAG

#### 1.6.1 Retrieval-Augmented Generation
- Principe et architecture
- Avantages pour les conseils agronomiques

#### 1.6.2 Bases de connaissances phytosanitaires
- Ressources IITA, FAO, CIRAD
- Structuration pour le RAG

### 1.7 Déploiement edge et mobile

#### 1.7.1 Quantification des modèles
- Techniques INT8/INT4
- Formats GGUF, ONNX, TFLite

#### 1.7.2 Plateformes de déploiement
- Serveurs cloud (API)
- Dispositifs edge (Jetson, Raspberry Pi)
- Applications mobiles (Android/iOS)

---

## CHAPITRE 2 : MATÉRIELS ET MÉTHODES

### 2.1 Environnement matériel et logiciel

#### 2.1.1 Configuration matérielle
- Station d'entraînement (RTX 5090, Intel Ultra 9, 64GB RAM)
- Dispositifs de déploiement cibles

#### 2.1.2 Environnement logiciel
- Système d'exploitation (Ubuntu 24.04)
- Frameworks (PyTorch, Transformers, PEFT)
- Outils de développement

### 2.2 Données

#### 2.2.1 Sources de données
- Dataset Kaggle Cassava Leaf Disease (21,367 images)
- Dataset IITA Tanzania
- Dataset Makerere University

#### 2.2.2 Analyse et préparation des données
- Distribution des classes
- Gestion du déséquilibre
- Partitionnement train/validation/test

#### 2.2.3 Génération du corpus multimodal
- Création des paires image-texte
- Templates de questions-réponses
- Traduction et adaptation française

#### 2.2.4 Augmentation de données
- Techniques géométriques et photométriques
- Augmentation spécifique aux VLM

### 2.3 Pipeline de prétraitement d'images

#### 2.3.1 Détection des feuilles avec GroundingDINO
- Configuration des prompts textuels
- Paramètres de seuils (box_threshold, text_threshold)

#### 2.3.2 Segmentation avec SAM-3
- Génération des masques
- Post-traitement des résultats

#### 2.3.3 Stratégies d'extraction
- Recadrage vs masquage
- Validation de la qualité

### 2.4 Architecture du modèle CassavaVLM

#### 2.4.1 Choix du modèle de base
- Justification de Qwen2.5-VL-7B
- Comparaison avec alternatives

#### 2.4.2 Configuration LoRA
- Modules cibles (q_proj, k_proj, v_proj, o_proj)
- Hyperparamètres (r=16, α=32, dropout=0.05)

#### 2.4.3 Architecture globale
- Encodeur visuel
- Module d'alignement
- Décodeur LLM

### 2.5 Protocole d'entraînement

#### 2.5.1 Phase 1 : Alignement du projecteur
- Configuration et objectifs
- Durée et métriques

#### 2.5.2 Phase 2 : Fine-tuning LoRA
- Hyperparamètres d'entraînement
- Scheduler et optimiseur
- Techniques de régularisation

#### 2.5.3 Stratégies d'optimisation
- Mixed precision (bfloat16)
- Gradient checkpointing
- Gradient accumulation

### 2.6 Système RAG pour les recommandations

#### 2.6.1 Construction de la base de connaissances
- Sources documentaires
- Chunking et indexation

#### 2.6.2 Pipeline de retrieval
- Modèle d'embedding (sentence-camembert-large)
- Base vectorielle (ChromaDB)

#### 2.6.3 Génération des recommandations
- Template de prompt
- Personnalisation contextuelle

### 2.7 Déploiement multi-plateforme

#### 2.7.1 API Cloud
- Architecture FastAPI
- Optimisations serveur

#### 2.7.2 Déploiement edge
- Quantification GGUF INT4
- Configuration Raspberry Pi / Jetson

#### 2.7.3 Application mobile Flutter
- Architecture offline-first
- Intégration TFLite
- Interface utilisateur

### 2.8 Métriques d'évaluation

#### 2.8.1 Métriques de classification
- Accuracy, Précision, Rappel, F1-score
- Matrice de confusion
- AUC-ROC

#### 2.8.2 Métriques de génération textuelle
- BLEU, ROUGE-L
- BERTScore
- Évaluation humaine

#### 2.8.3 Métriques de déploiement
- Latence (temps d'inférence)
- Consommation mémoire
- Tokens par seconde

---

## CHAPITRE 3 : RÉSULTATS ET ANALYSE

### 3.1 Résultats du prétraitement

#### 3.1.1 Performance de la segmentation
- Qualité des détections GroundingDINO
- Précision des masques SAM-3
- Exemples visuels

#### 3.1.2 Impact sur la classification
- Comparaison avec/sans prétraitement
- Analyse statistique

### 3.2 Performances de classification

#### 3.2.1 Résultats globaux
- Accuracy atteinte
- Métriques par fold (validation croisée)

#### 3.2.2 Analyse par classe
- F1-score par maladie
- Matrice de confusion détaillée
- Classes difficiles et raisons

#### 3.2.3 Comparaison avec les baselines
- Versus CNN (ConvNeXt, EfficientNet)
- Versus autres VLM (BLIP-2, CLIP-LoRA)
- Gains obtenus

### 3.3 Qualité des réponses textuelles

#### 3.3.1 Évaluation automatique
- Scores BLEU et ROUGE
- BERTScore

#### 3.3.2 Évaluation humaine
- Protocole d'évaluation
- Pertinence et exactitude
- Comparaison avec expertise agronomique

### 3.4 Performance du système RAG

#### 3.4.1 Qualité du retrieval
- Précision des documents récupérés
- Couverture des maladies

#### 3.4.2 Qualité des recommandations générées
- Pertinence agronomique
- Adaptation au contexte

### 3.5 Performances de déploiement

#### 3.5.1 Latence par plateforme
- Cloud API
- Dispositif edge
- Application mobile

#### 3.5.2 Consommation de ressources
- Mémoire GPU/RAM
- Taille des modèles quantifiés

#### 3.5.3 Mode offline
- Performances dégradées gracieuses
- Fiabilité du fallback

### 3.6 Analyse des erreurs

#### 3.6.1 Cas d'erreurs de classification
- Exemples visuels
- Causes identifiées

#### 3.6.2 Limites des réponses textuelles
- Hallucinations éventuelles
- Cas non couverts

---

## CHAPITRE 4 : DISCUSSION ET APPLICATION

### 4.1 Discussion des résultats

#### 4.1.1 Validation des hypothèses
- H1 : VLM vs CNN
- H2 : Support français
- H3 : Déploiement offline

#### 4.1.2 Comparaison avec l'état de l'art
- Positionnement par rapport aux travaux existants
- Apports du travail

#### 4.1.3 Limites de l'étude
- Contraintes du dataset
- Généralisation géographique
- Absence de validation terrain

### 4.2 Application développée : CassavaDoc

#### 4.2.1 Architecture du système
- Composants et flux de données
- Modes de fonctionnement

#### 4.2.2 Interface utilisateur
- Captures d'écran
- Parcours utilisateur type

#### 4.2.3 Fonctionnalités principales
- Diagnostic par photo
- Questions interactives en français
- Recommandations de traitement
- Historique et export PDF

### 4.3 Démonstration et cas d'usage

#### 4.3.1 Scénarios d'utilisation
- Agriculteur en champ
- Agent de vulgarisation
- Formation agricole

#### 4.3.2 Exemples de diagnostics
- Captures avec différentes maladies
- Réponses du système

### 4.4 Perspectives et travaux futurs

#### 4.4.1 Améliorations techniques
- Modèles plus légers
- Multi-tâches (sévérité, rendement)

#### 4.4.2 Extensions fonctionnelles
- Support des langues locales (Fon, Yoruba)
- Système d'alerte épidémiologique
- Intégration IoT

#### 4.4.3 Validation et déploiement
- Tests avec agriculteurs béninois
- Partenariats institutionnels (INRAB, IITA)

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

- Annexe A : Extraits de code source
- Annexe B : Configuration détaillée des hyperparamètres
- Annexe C : Exemples complets de dialogues avec le modèle
- Annexe D : Guide d'utilisation de l'application
- Annexe E : Questionnaire d'évaluation humaine

---

## TABLE DES MATIÈRES DÉTAILLÉE (estimations de pages)

| Section | Pages estimées |
|---------|----------------|
| Pages liminaires | 10-12 |
| Introduction générale | 5-7 |
| Chapitre 1 : Revue de littérature | 25-30 |
| Chapitre 2 : Matériels et méthodes | 20-25 |
| Chapitre 3 : Résultats et analyse | 20-25 |
| Chapitre 4 : Discussion et application | 15-20 |
| Conclusion générale | 3-4 |
| Références bibliographiques | 5-6 |
| Annexes | 10-15 |
| **TOTAL** | **115-145 pages** |