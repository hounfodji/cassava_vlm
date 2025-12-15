# INTRODUCTION GÉNÉRALE

Le manioc (*Manihot esculenta* Crantz) occupe une place singulière dans le paysage agricole mondial. Troisième source de glucides après le blé et le riz, cette plante tropicale assure quotidiennement la subsistance de plus de 800 millions de personnes à travers le monde (FAO, 2023). Selon les données de FAOSTAT, la production mondiale a atteint 333,7 millions de tonnes en 2023, marquant une progression de 32% par rapport à 2010. En Afrique subsaharienne, où se concentre 63,6% de cette production (Osabohien et al., 2023), le manioc dépasse le simple statut de culture vivrière pour devenir un véritable pilier de la sécurité alimentaire. Sa rusticité face aux sols pauvres et aux épisodes de sécheresse en fait une culture de résilience, particulièrement précieuse dans un contexte de changements climatiques (Jarvis et al., 2012).

Au Bénin, cette réalité prend une dimension encore plus marquée. Avec une production annuelle avoisinant 4,4 millions de tonnes selon FAOSTAT (2023), le manioc constitue la première culture du pays en volume. Le secteur agricole contribue à 26,9% du PIB national et emploie plus de 70% de la population active (Banque Mondiale, 2019). Les racines transformées en gari, tapioca ou lafun nourrissent des millions de familles, tandis que la filière fait vivre environ 550 000 exploitations agricoles, majoritairement familiales et d'une superficie moyenne de 1,7 hectare (MAEP, 2021). La consommation nationale de gari atteint 126 kilogrammes par personne et par an (Aréolis, 2023), témoignant de l'ancrage profond de cette culture dans les habitudes alimentaires béninoises.

Pourtant, cette dépendance au manioc expose les populations à une vulnérabilité considérable lorsque les maladies frappent les cultures. La mosaïque du manioc (CMD), la striure brune (CBSD), la bactériose (CBB) et les attaques d'acariens verts (CGM) causent chaque année des pertes estimées entre 1,9 et 2,7 milliards de dollars américains à l'échelle du continent africain (Patil et Fauquet, 2009 ; Legg et al., 2014). Au Bénin, les enquêtes de terrain menées par le programme WAVE en 2020 révèlent que 81% des champs de manioc présentent des symptômes de mosaïque, avec des taux atteignant 100% dans certaines régions comme l'Ouémé et l'Alibori (Zinsou et al., 2022). Ces chiffres traduisent une situation sanitaire préoccupante qui menace directement la sécurité alimentaire des 3,3 millions de Béninois considérés comme vulnérables sur le plan nutritionnel (PAM, 2022).

Face à cette menace, le diagnostic précoce et fiable des maladies apparaît comme un levier essentiel de lutte. Cependant, les méthodes conventionnelles se heurtent à des obstacles majeurs. L'inspection visuelle par les agriculteurs eux-mêmes ne permet d'identifier correctement les maladies que dans 18 à 31% des cas, contre 40 à 58% pour les agents de vulgarisation formés (Ramcharan et al., 2019). Les analyses en laboratoire, bien que précises, restent inaccessibles à la grande majorité des petits producteurs : un test PCR complet coûte plus de 120 dollars américains et nécessite des équipements que peu de structures possèdent en zone rurale (Bisimwa et al., 2022). Le ratio d'experts phytopathologistes par rapport à la population avoisine 1 pour 1 million en Afrique, contre 1 pour 25 000 aux États-Unis (Wilson et al., 2018), rendant illusoire tout espoir de diagnostic expert généralisé.

L'intelligence artificielle, et plus particulièrement l'apprentissage profond, a ouvert ces dernières années des perspectives nouvelles pour démocratiser le diagnostic phytosanitaire. Des applications comme PlantVillage Nuru, développée conjointement par Penn State University et l'IITA, ou Plantix de PEAT GmbH, permettent désormais aux agriculteurs équipés de smartphones d'obtenir un diagnostic instantané en photographiant simplement les feuilles de leurs plants. Ces outils reposent sur des réseaux de neurones convolutifs (CNN) entraînés à reconnaître les symptômes caractéristiques de chaque maladie. Sur le manioc, PlantVillage Nuru atteint une précision de 74 à 88% en conditions réelles lorsque six feuilles par plant sont analysées (Mrisho et al., 2020).

Malgré ces avancées prometteuses, l'adoption de ces technologies demeure limitée. Une étude récente conduite par Ahoya et al. (2024) au Bénin montre que seulement 14,1% des agriculteurs interrogés utilisent effectivement l'application PlantVillage Nuru. Plusieurs facteurs expliquent cette faible appropriation. D'une part, les applications actuelles fonctionnent comme des boîtes noires qui délivrent un diagnostic sans l'expliquer, ce qui n'inspire pas confiance aux utilisateurs (Mohanty et al., 2016). D'autre part, elles ne permettent aucune interaction : l'agriculteur ne peut pas poser de questions complémentaires ni obtenir des conseils adaptés à sa situation particulière. Les recommandations de traitement restent génériques, ignorant les ressources localement disponibles et les contraintes économiques des petits producteurs. Enfin, le support linguistique se limite généralement au français ou à l'anglais, alors que la majorité des agriculteurs béninois communiquent en fon, yoruba ou dans d'autres langues nationales qui représentent la langue maternelle de 65% de la population (Recensement Général de la Population, 2013).

Une nouvelle génération de modèles d'intelligence artificielle, les Vision-Language Models (VLM), offre aujourd'hui la possibilité de dépasser ces limitations. Ces architectures, dont BLIP-2 (Li et al., 2023) et Qwen2-VL (Bai et al., 2024) constituent des exemples représentatifs, combinent la capacité d'analyser des images avec celle de comprendre et de générer du langage naturel. Elles permettent ainsi d'établir un véritable dialogue avec l'utilisateur : le modèle peut expliquer son diagnostic, répondre aux questions, demander des précisions et formuler des recommandations personnalisées. Des travaux récents, notamment ceux de Kaur et al. (2025) sur le système TLDVLM pour la tomate, ont démontré l'efficacité de cette approche avec une précision atteignant 97,27% tout en offrant des capacités d'interaction en langage naturel.

À ce jour, et malgré une recherche bibliographique approfondie dans les bases Scopus, Web of Science et Google Scholar, aucun Vision-Language Model n'a été développé spécifiquement pour le diagnostic des maladies du manioc. Cette lacune constitue l'opportunité de recherche que le présent mémoire se propose d'explorer.


## 1. Problématique

Les systèmes actuels de diagnostic des maladies du manioc par intelligence artificielle présentent des limitations structurelles qui freinent leur adoption par les agriculteurs africains. Ces outils, fondés sur des architectures de réseaux de neurones convolutifs, se contentent de produire une classification binaire ou multiclasse sans fournir d'explication sur le raisonnement qui sous-tend le diagnostic (Sambasivam et Opiyo, 2021). L'agriculteur reçoit une étiquette — mosaïque, bactériose, feuille saine — sans comprendre quels symptômes ont conduit à cette conclusion ni quelle confiance accorder au résultat.

Cette opacité pose un problème de confiance particulièrement aigu dans le contexte agricole béninois. Les agriculteurs, habitués à s'appuyer sur leur expérience et sur les conseils de leurs pairs, peinent à accepter les verdicts d'une application qu'ils ne comprennent pas (Fabregas et al., 2019). Lorsque le diagnostic contredit leur intuition, ils tendent à l'ignorer plutôt qu'à remettre en question leur jugement initial. L'absence de dialogue empêche toute forme de pédagogie qui permettrait progressivement de construire cette confiance.

Par ailleurs, 93,51% des agriculteurs béninois utilisent des boutures provenant de leurs propres champs ou de champs voisins pour leurs nouvelles plantations (Ahoya et al., 2024), ignorant que ce vecteur constitue le principal mode de propagation des virus. Un système de diagnostic capable d'expliquer cette relation causale et de recommander des sources de matériel végétal sain contribuerait significativement à briser ce cycle de contamination.

Au-delà du diagnostic lui-même, les agriculteurs ont besoin de conseils actionnables pour protéger leurs cultures. Quelle variété résistante planter la saison prochaine ? Quel traitement appliquer avec les moyens disponibles localement ? Comment éviter la propagation aux parcelles voisines ? Les applications existantes n'apportent que des réponses génériques, déconnectées des réalités du terrain béninois (Adenle et al., 2019). Un agriculteur du Couffo et un autre de l'Alibori n'ont pas accès aux mêmes intrants, ne cultivent pas les mêmes variétés et ne font pas face aux mêmes conditions pédoclimatiques. Une recommandation pertinente doit tenir compte de ce contexte.

La barrière linguistique constitue un obstacle supplémentaire non négligeable. Si le français demeure la langue officielle du Bénin, seuls 35% environ de la population le maîtrisent réellement, et ce taux chute drastiquement en milieu rural où le taux d'alphabétisation oscille entre 30 et 40% (UNESCO, 2021). Les langues nationales comme le fon (24% de la population), le yoruba (12%), l'aja (15%) ou le bariba (9,6%) dominent la communication quotidienne des agriculteurs (Ethnologue, 2023). Une application qui ne s'exprime qu'en français ou en anglais exclut de facto la majorité de son public cible.

Enfin, la connectivité internet reste précaire dans les zones rurales béninoises. Bien que la couverture mobile progresse rapidement, avec un taux de pénétration de 67,3% et 8,5 millions d'abonnés uniques en 2023 (ARCEP Bénin, 2023), l'accès effectif à internet ne concerne que 33,8% de la population selon DataReportal (2024). Le rapport de la GSMA (2024) sur l'Afrique subsaharienne souligne que 60% des personnes couvertes par un réseau mobile n'utilisent pas l'internet mobile, principalement pour des raisons de coût et d'accessibilité. Les applications nécessitant une connexion permanente pour fonctionner se trouvent donc inutilisables dans de nombreuses situations où le diagnostic serait pourtant nécessaire.

La question centrale qui guide ce travail peut donc se formuler ainsi : **comment concevoir un système de diagnostic des maladies du manioc qui soit à la fois précis, explicable, interactif, adapté au contexte linguistique béninois et fonctionnel en conditions de connectivité limitée ?**


## 2. Hypothèses de recherche

Pour répondre à cette problématique, trois hypothèses de recherche orientent ce travail.

**Hypothèse 1** : Un Vision-Language Model adapté au domaine agricole, combinant un encodeur visuel performant avec un grand modèle de langage multilingue, peut atteindre une précision de classification supérieure à celle des approches CNN traditionnelles sur les maladies du manioc, tout en offrant des capacités d'explication et d'interaction en langage naturel.

Cette hypothèse s'appuie sur les résultats obtenus par Kaur et al. (2025) qui ont démontré un gain de 2,7 points de pourcentage du système TLDVLM par rapport aux meilleurs CNN sur la tomate (97,27% contre 94,57% pour ConvNeXt-tiny). Elle postule que ce gain peut être reproduit voire amplifié sur le manioc grâce à la capacité des VLM à exploiter conjointement l'information visuelle et textuelle.

**Hypothèse 2** : L'utilisation d'un modèle de langage nativement multilingue comme Qwen2.5-VL, combinée à une stratégie de fine-tuning efficace par LoRA (Low-Rank Adaptation), permet de développer un système capable de dialoguer en français avec les agriculteurs sans dégradation significative des performances de classification.

Cette hypothèse repose sur les travaux de Hu et al. (2022) qui ont montré que LoRA permet d'adapter efficacement de grands modèles à de nouveaux domaines avec moins de 1% de paramètres entraînables, ainsi que sur les capacités multilingues documentées de Qwen2.5-VL couvrant plus de 32 langues dont le français (Bai et al., 2024).

**Hypothèse 3** : Les techniques de quantification (INT4/INT8) et d'optimisation permettent de déployer un Vision-Language Model sur des dispositifs à ressources limitées (smartphones Android, Raspberry Pi) tout en maintenant une qualité de diagnostic acceptable pour un usage en conditions réelles de terrain.

Cette hypothèse s'inscrit dans la continuité des travaux sur les modèles embarqués comme MobileVLM (Chu et al., 2024) et des démonstrations de déploiement edge réalisées par Kaur et al. (2025) sur Raspberry Pi 5.


## 3. Objectifs

### 3.1. Objectif général

L'objectif général de ce travail est de concevoir, développer et évaluer le premier Vision-Language Model dédié au diagnostic interactif des maladies du manioc, capable de fournir des explications et des recommandations de traitement personnalisées en français, avec possibilité de fonctionnement hors ligne.

### 3.2. Objectifs spécifiques

Pour atteindre cet objectif général, cinq objectifs spécifiques ont été définis :

**OS1** : Constituer un corpus multimodal de qualité pour le fine-tuning du VLM, en combinant les datasets existants (Kaggle Cassava Leaf Disease, IITA Tanzania, Makerere University) avec des paires question-réponse générées en français couvrant le diagnostic, les symptômes et les recommandations de traitement.

**OS2** : Développer un pipeline de prétraitement d'images intégrant la détection automatique des feuilles par GroundingDINO et leur segmentation précise par SAM-3, afin d'isoler les régions d'intérêt avant classification.

**OS3** : Adapter le modèle Qwen2.5-VL-7B au domaine du manioc par fine-tuning LoRA, en optimisant les hyperparamètres (rang, alpha, modules cibles) pour maximiser les performances de classification et la qualité des réponses en français.

**OS4** : Concevoir et implémenter un système de génération de recommandations basé sur l'approche RAG (Retrieval-Augmented Generation), s'appuyant sur une base de connaissances phytosanitaires structurée à partir des ressources de l'IITA, de la FAO et du CIRAD.

**OS5** : Développer une stratégie de déploiement hybride comprenant une API cloud pour les contextes connectés, une version quantifiée pour dispositifs edge (Raspberry Pi, Jetson) et une application mobile Flutter fonctionnant en mode hors ligne.


## 4. Méthodologie résumée

La démarche méthodologique adoptée pour ce travail s'articule en quatre phases principales, suivant une approche expérimentale itérative.

La première phase porte sur la collecte et la préparation des données. Elle consiste à rassembler les images provenant de trois sources principales : le dataset Kaggle Cassava Leaf Disease Competition comprenant 21 367 images annotées (Mwebaze et al., 2019), le dataset IITA Tanzania avec 15 000 images incluant des annotations de sévérité, et le dataset Makerere University offrant 9 103 images avec boîtes englobantes. Ces données brutes sont ensuite traitées par le pipeline de segmentation GroundingDINO + SAM-3 pour isoler les feuilles individuelles. Un corpus multimodal est constitué en générant des paires question-réponse en français à partir des annotations de classe, couvrant différents types d'interactions : identification de maladie, description des symptômes, explication du diagnostic et recommandations de traitement.

La deuxième phase concerne le développement et l'entraînement du modèle. Le choix architectural se porte sur Qwen2.5-VL-7B, un Vision-Language Model de 7 milliards de paramètres offrant un support natif du français et des performances de pointe sur les benchmarks visuels. L'adaptation au domaine du manioc s'effectue par fine-tuning LoRA avec une configuration optimisée (rang r=16, alpha=32, dropout=0,05) ciblant les couches d'attention du modèle. L'entraînement se déroule sur une station équipée d'un GPU NVIDIA RTX 5090 (32 Go VRAM), d'un processeur Intel Core Ultra 9 285K et de 64 Go de RAM, sous Ubuntu 24.04. Le protocole d'entraînement comprend une phase d'alignement du projecteur vision-langage suivie d'un fine-tuning LoRA sur 3 époques avec validation croisée stratifiée à 5 plis.

La troisième phase est dédiée à l'intégration du système RAG pour la génération de recommandations. Une base de connaissances phytosanitaires est constituée à partir des guides techniques de l'IITA (Disease Control in Cassava Farms: IPM Field Guide), des ressources FAO (Save and Grow: Cassava) et des publications du CIRAD. Les documents sont segmentés en chunks sémantiques, vectorisés à l'aide du modèle sentence-camembert-large optimisé pour le français, et indexés dans une base ChromaDB. Le système RAG permet ainsi d'enrichir les réponses du VLM avec des informations actualisées et contextuellement pertinentes.

La quatrième phase porte sur le déploiement et l'évaluation. Trois configurations de déploiement sont développées : une API cloud FastAPI pour l'inférence haute qualité, une version quantifiée GGUF INT4 pour les dispositifs edge type Raspberry Pi 5, et une application mobile Flutter intégrant un modèle TFLite pour le fonctionnement hors ligne. L'évaluation s'appuie sur des métriques de classification (accuracy, F1-score, AUC-ROC), des métriques de génération textuelle (BLEU, ROUGE-L, BERTScore) et des métriques de déploiement (latence, consommation mémoire). Une évaluation humaine sur un échantillon de 500 diagnostics complète l'évaluation automatique.


## 5. Structure du mémoire

Le présent mémoire s'organise en quatre chapitres, encadrés par cette introduction générale et une conclusion.

Le **premier chapitre** présente une revue de la littérature structurée en plusieurs volets. Il expose d'abord l'importance économique et alimentaire du manioc ainsi que les principales maladies qui l'affectent. Il dresse ensuite un état de l'art des méthodes de diagnostic par intelligence artificielle, des réseaux de neurones convolutifs aux Vision-Language Models. Les techniques d'adaptation efficace des grands modèles (LoRA, QLoRA), les approches de segmentation d'images (GroundingDINO, SAM-2) et les systèmes de génération augmentée par récupération (RAG) sont également examinés. Ce chapitre se conclut par l'identification des lacunes dans la littérature qui justifient la présente recherche.

Le **deuxième chapitre** détaille les matériels et méthodes mobilisés. Il décrit l'environnement matériel et logiciel, les sources de données utilisées et le processus de constitution du corpus multimodal. L'architecture du modèle CassavaVLM est présentée en détail, de même que le protocole d'entraînement, le système RAG et les différentes configurations de déploiement. Les métriques d'évaluation retenues sont justifiées au regard des objectifs de recherche.

Le **troisième chapitre** expose les résultats obtenus à chaque étape du travail : performances du pipeline de prétraitement, résultats de classification comparés aux baselines CNN, qualité des réponses textuelles évaluée automatiquement et par des experts, efficacité du système RAG, et performances de déploiement sur les différentes plateformes cibles. Une analyse approfondie des erreurs complète cette présentation.

Le **quatrième chapitre** propose une discussion des résultats au regard des hypothèses formulées et de l'état de l'art. Il présente l'application développée, baptisée CassavaDoc, à travers des scénarios d'utilisation concrets. Les limites du travail sont honnêtement discutées et des perspectives de recherche et de développement sont tracées pour les travaux futurs.

La **conclusion générale** synthétise les contributions du mémoire, rappelle les principaux résultats et ouvre sur les implications pratiques pour l'agriculture béninoise et africaine.