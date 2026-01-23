# INTRODUCTION GÉNÉRALE

## 1. Contexte et justification

Le manioc (*Manihot esculenta* Crantz) constitue la troisième source mondiale de glucides après le riz et le maïs, et représente un pilier fondamental de la sécurité alimentaire dans les régions tropicales [1]. Selon les données les plus récentes de la FAO, la production mondiale de manioc a atteint 333,68 millions de tonnes en 2023, marquant une croissance soutenue de 32% par rapport aux 252 millions de tonnes enregistrées en 2010 [2]. Cette progression remarquable témoigne de l'importance croissante de cette culture dans les systèmes alimentaires mondiaux, particulièrement face aux défis posés par les changements climatiques.

L'Afrique subsaharienne domine la production mondiale avec une contribution de 56 à 63% du volume total, soit environ 186 à 200 millions de tonnes annuellement [2, 3]. Le Nigeria occupe la première place mondiale avec 62,7 millions de tonnes, suivi de la République Démocratique du Congo (45,2 millions de tonnes) et du Ghana (26,5 millions de tonnes) [2]. Cette prédominance africaine s'explique par la résilience exceptionnelle du manioc face aux conditions climatiques difficiles et aux sols pauvres, lui valant le surnom de « culture de la famine » dans de nombreuses communautés rurales [4].

Le manioc assure la nutrition quotidienne de plus de 700 millions de personnes en Afrique, représentant environ 37% de l'apport calorique du continent [3, 5]. Cette dépendance nutritionnelle est particulièrement marquée en Afrique centrale, où la consommation peut dépasser 1000 calories par personne et par jour en République Démocratique du Congo [5]. Au-delà de son rôle alimentaire, le manioc génère des revenus substantiels pour des millions de petits agriculteurs et alimente une chaîne de valeur diversifiée incluant la transformation en gari, tapioca, amidon industriel et bioéthanol [6].

Le Bénin se classe au 19ème rang mondial des producteurs de manioc avec une production annuelle de 4,4 millions de tonnes en 2023, représentant environ 1,3% de la production mondiale et le 7ème rang africain [2]. Cette production fait du manioc la première culture vivrière du pays en termes de tonnage, devant le maïs et l'igname. Les départements du Zou, des Collines et de l'Atlantique concentrent historiquement environ 75% de la production nationale, bénéficiant de conditions pédoclimatiques favorables avec des sols ferralitiques et une pluviométrie comprise entre 1000 et 1400 millimètres par an [7].

*(Insérer Figure 1 : Carte de la production mondiale de manioc montrant la dominance de l'Afrique subsaharienne et la position du Bénin)*

**Figure 1** : Répartition géographique de la production mondiale de manioc en 2023. L'Afrique subsaharienne représente 56-63% de la production mondiale, avec le Bénin au 19ème rang mondial (Source : adapté de FAOSTAT, 2024).

Cependant, cette culture stratégique fait face à des menaces phytosanitaires majeures qui compromettent gravement la sécurité alimentaire et les moyens de subsistance de millions d'agriculteurs. Les maladies virales, principalement la mosaïque du manioc (CMD - *Cassava Mosaic Disease*) et la striure brune du manioc (CBSD - *Cassava Brown Streak Disease*), causent des pertes économiques estimées entre 1,9 et 2,7 milliards USD par an en Afrique [8, 9]. La CMD, transmise par l'aleurode *Bemisia tabaci* et par les boutures infectées, provoque une réduction moyenne des rendements de 15 à 24% à l'échelle africaine, pouvant atteindre 50 à 100% dans les zones fortement infectées ou pour les variétés hautement sensibles [10, 11].

**Tableau 1** : Impact économique des principales maladies du manioc en Afrique

| Maladie | Pertes annuelles (USD) | Réduction rendement | Zones principales |
|---------|----------------------|---------------------|-------------------|
| CMD (Mosaïque) | 1,9 - 2,7 milliards | 15-24% (moyenne), jusqu'à 100% | Afrique entière |
| CBSD (Striure brune) | 736 - 750 millions | Jusqu'à 70% | Afrique de l'Est |
| CBB (Bactériose) | Non quantifié | 20-100% | Zones humides |
| CGM (Acarien vert) | Non quantifié | 13-80% | Afrique de l'Est/Centrale |

*Sources : Patil & Fauquet (2009) [8], IITA (2014) [9], Thresh et al. (1997) [11]*

Au Bénin, la situation est particulièrement préoccupante. Une étude récente de Houngue et al. (2022) révèle que 80,55% des champs de manioc sont infectés par le CMD (145 sur 180 champs enquêtés dans 11 régions), avec une incidence moyenne de 34% et une sévérité de 2,85 sur une échelle de 1 à 5 [12]. Les régions d'Ouémé et d'Alibori présentent une prévalence de 100%, avec certaines localités comme Malanville atteignant une incidence totale [12]. Ces chiffres alarmants soulignent l'urgence de développer des outils de diagnostic accessibles et efficaces pour permettre aux agriculteurs d'identifier précocement les maladies et de prendre des mesures de gestion appropriées.

Face à ces défis, les méthodes traditionnelles de diagnostic reposant sur l'expertise de phytopathologistes se heurtent à des contraintes structurelles majeures. Le ratio agents de vulgarisation agricole par agriculteur en Afrique subsaharienne est catastrophique : 1 agent pour 1 500 agriculteurs en moyenne, alors que la FAO recommande un ratio de 1:400 à 1:500 [13]. Au Nigeria, premier producteur mondial, ce ratio atteint 1:5 000 voire 1:10 000, soit 3 à 25 fois inférieur aux normes internationales [14]. Cette pénurie critique empêche la dissémination efficace des connaissances phytosanitaires et des technologies de diagnostic auprès des communautés rurales.

L'émergence de l'intelligence artificielle, et plus particulièrement du deep learning, a ouvert de nouvelles perspectives pour l'automatisation du diagnostic des maladies végétales. Depuis les travaux pionniers de Ramcharan et al. (2017) démontrant l'applicabilité des réseaux de neurones convolutifs (CNN) au diagnostic du manioc avec une précision de 93% sur smartphone [15], de nombreuses architectures ont été développées. Les modèles les plus récents, tels que MiXceptionLeaf (98,8%) et CDDNet (98,95%), atteignent des performances remarquables sur les benchmarks de laboratoire [16, 17].

L'application PlantVillage Nuru, développée par Penn State University en partenariat avec l'IITA, représente le déploiement le plus significatif de ces technologies en Afrique, touchant plus de 50 000 agriculteurs dans 19 pays [18]. Cependant, des évaluations rigoureuses en conditions réelles révèlent des limitations importantes. Mrisho et al. (2020) démontrent que la précision de Nuru chute drastiquement pour les symptômes légers : seulement 37-43% pour le CMD léger et 13-27% pour le CBSD [19]. En mode vidéo temps réel, la précision tombe à 29,4% pour les symptômes légers [19]. Le score F-1 subit une baisse de 32% entre les données de test en laboratoire et les conditions terrain [20].

Au-delà des performances, les approches CNN présentent des limitations fondamentales qui entravent leur adoption par les agriculteurs :

- **Absence d'explicabilité** : Les CNN fonctionnent en « boîte noire », fournissant un diagnostic sans expliquer le raisonnement sous-jacent, ce qui réduit la confiance des utilisateurs [21].
- **Manque d'interactivité** : Les systèmes actuels ne permettent pas aux agriculteurs de poser des questions complémentaires sur les symptômes observés, les causes possibles ou les traitements recommandés.
- **Support linguistique limité** : Bien que Nuru supporte le français, l'anglais, le swahili et le twi, les langues locales béninoises (Fon, Yoruba, Dendi) restent exclues, créant une barrière à l'adoption [22].
- **Conseils statiques** : Les recommandations de traitement sont pré-écrites par des experts et non générées contextuellement en fonction de la situation spécifique de l'agriculteur.

Une étude récente au Bénin (Ahoya et al., 2024) confirme ces obstacles : le taux d'adoption de l'application Nuru n'est que de 14,1%, limité par la non-disponibilité d'appareils ICT (79,6% des répondants), le coût des smartphones (72,9%) et le niveau d'éducation requis [22]. Ces résultats suggèrent qu'une nouvelle génération d'outils, plus interactifs, explicatifs et linguistiquement accessibles, est nécessaire pour répondre aux besoins réels des agriculteurs africains.

Les Vision-Language Models (VLM), une classe émergente de modèles d'intelligence artificielle combinant la compréhension visuelle et le traitement du langage naturel, offrent une opportunité unique de surmonter ces limitations. Contrairement aux CNN qui se limitent à la classification, les VLM peuvent analyser une image tout en engageant un dialogue en langage naturel avec l'utilisateur [23]. Cette capacité multimodale permet d'expliquer le diagnostic, de répondre à des questions de suivi et de générer des recommandations contextualisées.

Les avancées récentes dans ce domaine sont remarquables. Le modèle LLaVA-PlantDiag (2024) atteint 96% de précision sur le diagnostic des maladies végétales tout en offrant des capacités d'explication conversationnelle [24]. Agri-LLaVA, fine-tuné sur plus de 400 000 échantillons agricoles couvrant 221 types de ravageurs et maladies, démontre la faisabilité d'assistants agricoles intelligents [25]. Le benchmark AgroBench (ICCV 2025), premier standard d'évaluation des VLM agricoles, établit 682 catégories de maladies sur 203 types de cultures [26].

Cependant, malgré ces progrès significatifs, **aucun Vision-Language Model n'a été développé spécifiquement pour le diagnostic des maladies du manioc**. Cette lacune est particulièrement critique compte tenu de l'importance de cette culture pour la sécurité alimentaire africaine. Les travaux existants sur le manioc utilisent exclusivement des approches CNN sans capacités conversationnelles ni explicatives [15, 20, 27].

Par ailleurs, la démocratisation des techniques de fine-tuning efficace, notamment LoRA (*Low-Rank Adaptation*) et QLoRA, rend désormais accessible l'adaptation de grands modèles multimodaux avec des ressources computationnelles limitées [28, 29]. Le modèle Qwen2.5-VL-7B, publié en janvier 2025, offre un support multilingue natif incluant le français, des performances rivalisant avec GPT-4o sur de nombreux benchmarks, et peut être fine-tuné avec seulement 7 GB de mémoire GPU grâce à QLoRA [30]. Ces avancées technologiques créent une fenêtre d'opportunité pour développer un assistant de diagnostic du manioc véritablement adapté au contexte africain francophone.

## 2. Problématique

Dans ce contexte, la problématique centrale de ce travail peut être formulée ainsi :

**Comment développer un système de diagnostic des maladies du manioc qui dépasse les limitations des approches CNN actuelles en offrant des capacités d'explication, d'interaction conversationnelle et de génération de conseils agronomiques personnalisés en français, tout en maintenant des performances de classification compétitives ?**

Cette problématique soulève plusieurs questions de recherche spécifiques :

1. Un Vision-Language Model fine-tuné peut-il atteindre des performances de classification comparables aux CNN spécialisés sur les maladies du manioc ?

2. Comment structurer un processus de fine-tuning multi-stage permettant au modèle d'acquérir à la fois des compétences de classification précise et des capacités de dialogue interactif ?

3. Le support multilingue natif de Qwen2.5-VL permet-il de générer des explications et recommandations de qualité en français pour les agriculteurs francophones ?

4. Quelles sont les contraintes techniques pour un éventuel déploiement de ce type de système dans le contexte africain caractérisé par une connectivité limitée ?

## 3. Hypothèses de recherche

Pour répondre à cette problématique, nous formulons les hypothèses suivantes :

**Hypothèse 1 (H1)** : Un Vision-Language Model (Qwen2.5-VL-7B) fine-tuné avec la technique LoRA peut atteindre une précision de classification des maladies du manioc supérieure à 75%, comparable aux performances des CNN de référence.

**Hypothèse 2 (H2)** : Une approche de fine-tuning multi-stage, combinant d'abord l'apprentissage de la classification puis l'instruction tuning pour le dialogue, permet d'obtenir un modèle capable à la fois de diagnostiquer avec précision et d'interagir de manière conversationnelle.

**Hypothèse 3 (H3)** : Le support multilingue natif de Qwen2.5-VL permet de générer des réponses de qualité en français sans dégradation significative des performances par rapport à l'anglais.

## 4. Objectifs

### 4.1 Objectif général

L'objectif général de ce travail est de concevoir, développer et évaluer **CassavaVLM**, le premier Vision-Language Model multilingue interactif dédié au diagnostic des maladies du manioc, capable de classifier les maladies, d'expliquer ses diagnostics et de fournir des conseils agronomiques personnalisés en français.

### 4.2 Objectifs spécifiques

Pour atteindre cet objectif général, les objectifs spécifiques suivants sont définis :

1. **Réaliser une synthèse bibliographique approfondie** sur les maladies du manioc, les approches de deep learning pour leur diagnostic, et les Vision-Language Models appliqués à l'agriculture.

2. **Développer un pipeline de fine-tuning multi-stage** de Qwen2.5-VL-7B utilisant la technique LoRA, comprenant :
   - Stage 1 : Fine-tuning pour la classification des 5 catégories (CMD, CBSD, CBB, CGM, Healthy)
   - Stage 2 : Instruction tuning pour les capacités de Visual Question Answering et la génération de conseils en français

3. **Construire un dataset d'instructions** en français pour l'entraînement des capacités conversationnelles du modèle, incluant des questions de diagnostic, des demandes d'explication et des requêtes de conseils agronomiques.

4. **Évaluer rigoureusement les performances** du modèle développé en termes de :
   - Précision de classification (accuracy, F1-score par classe)
   - Qualité des réponses générées (métriques automatiques et évaluation humaine)
   - Comparaison avec les approches CNN existantes

5. **Analyser la faisabilité du déploiement** en étudiant les options de quantification et les contraintes techniques pour une utilisation en contexte africain.

## 5. Méthodologie résumée

Pour atteindre ces objectifs, nous adoptons une méthodologie structurée en plusieurs phases :

**Phase 1 - Revue de littérature** : Analyse approfondie de l'état de l'art sur les maladies du manioc, les techniques de deep learning pour le diagnostic végétal, les architectures VLM récentes et les méthodes de fine-tuning efficace (LoRA, QLoRA).

**Phase 2 - Préparation des données** : Utilisation du dataset Kaggle Cassava Leaf Disease (21 367 images, 5 classes) avec stratégies d'oversampling pour les classes minoritaires (CBSD ×3, Healthy ×2) et augmentation de données via Albumentations.

**Phase 3 - Fine-tuning Stage 1 (Classification)** : Adaptation de Qwen2.5-VL-7B-Instruct avec LoRA (r=8, α=8) sur la tâche de classification multiple-choice. Entraînement sur GPU RTX 5090 avec optimisations mémoire (gradient checkpointing, mixed precision bf16).

**Phase 4 - Construction du dataset d'instructions** : Génération de paires question-réponse en français couvrant le diagnostic, l'explication des symptômes et les recommandations de traitement, validées par expertise agronomique.

**Phase 5 - Fine-tuning Stage 2 (VQA)** : Continuation du fine-tuning sur le dataset d'instructions pour développer les capacités conversationnelles tout en préservant les compétences de classification acquises au Stage 1.

**Phase 6 - Évaluation et analyse** : Évaluation quantitative (accuracy, F1-score, BLEU, BERTScore) et qualitative (évaluation humaine) des performances. Analyse comparative avec les baselines CNN et étude de la faisabilité de déploiement.

*(Insérer Figure 2 : Schéma méthodologique global montrant les différentes phases du projet)*

**Figure 2** : Méthodologie de développement de CassavaVLM illustrant l'approche de fine-tuning multi-stage : le Stage 1 développe les capacités de classification sur le dataset Kaggle, puis le Stage 2 ajoute les capacités conversationnelles via instruction tuning.

## 6. Structure du mémoire

Ce mémoire est organisé en trois chapitres, en plus de la présente introduction et de la conclusion générale :

**Le Chapitre 1 (Synthèse bibliographique)** présente une revue exhaustive de la littérature couvrant trois axes principaux : (i) l'importance du manioc et les défis phytosanitaires auxquels il fait face, (ii) l'évolution des approches de deep learning pour le diagnostic des maladies végétales, des CNN aux Vision Transformers, et (iii) l'émergence des Vision-Language Models et leur application en agriculture. Ce chapitre établit également les fondements théoriques des techniques de fine-tuning efficace (LoRA, QLoRA) et identifie les lacunes de la littérature justifiant notre contribution.

**Le Chapitre 2 (Matériels et méthodes)** détaille l'environnement expérimental (matériel et logiciel), les données utilisées et leur préparation, l'architecture du modèle CassavaVLM basé sur Qwen2.5-VL-7B, le protocole de fine-tuning multi-stage, la construction du dataset d'instructions en français, et les métriques d'évaluation retenues.

**Le Chapitre 3 (Résultats et discussions)** présente et analyse les résultats obtenus aux deux stages de fine-tuning. Les performances de classification sont comparées aux approches CNN existantes, et la qualité des réponses conversationnelles est évaluée par des métriques automatiques et une évaluation humaine. Ce chapitre discute également les implications pratiques des résultats, les limites de l'étude et les perspectives de déploiement.

La **conclusion générale** synthétise les contributions principales de ce travail, notamment le développement du premier VLM dédié au manioc, et propose des perspectives pour les travaux futurs incluant l'extension aux langues locales et la validation terrain avec des agriculteurs béninois.

---

## Références de l'Introduction

[1] Food and Agriculture Organization. (2024). *The State of Food and Agriculture 2024*. Rome: FAO.

[2] FAOSTAT. (2025). *Crops and livestock products database*. Rome: FAO. https://www.fao.org/faostat/en/#data/QCL

[3] Yalley, M.E., Donou-Hounsou, F., Rainer, J., & Azzam, O. (2020). CassavaMap, a fine-resolution disaggregation of cassava production and harvested area in Africa in 2014. *Scientific Data*, 7, 167. DOI: 10.1038/s41597-020-0501-z

[4] Jarvis, A., Ramirez-Villegas, J., Campo, B.V.H., & Navarro-Racines, C. (2012). Is cassava the answer to African climate change adaptation? *Tropical Plant Biology*, 5(1), 9-29.

[5] IITA. (2024). *Cassava - Crops*. International Institute of Tropical Agriculture, Ibadan. https://www.iita.org/cropsnew/cassava/

[6] Nweke, F.I., Spencer, D.S., & Lynam, J.K. (2002). *The cassava transformation: Africa's best-kept secret*. Michigan State University Press.

[7] FAO. (2005). *A review of cassava in Africa with country case studies on Nigeria, Ghana, the United Republic of Tanzania, Uganda and Benin*. Rome: FAO.

[8] Patil, B.L., & Fauquet, C.M. (2009). Cassava mosaic geminiviruses: actual knowledge and perspectives. *Molecular Plant Pathology*, 10(5), 685-701. DOI: 10.1111/j.1364-3703.2009.00559.x

[9] IITA. (2014). *Cassava brown streak disease: A threat to food security in Africa*. Ibadan: IITA.

[10] Legg, J.P., & Thresh, J.M. (2000). Cassava mosaic virus disease in East Africa: a dynamic disease in a changing environment. *Virus Research*, 71(1-2), 135-149.

[11] Thresh, J.M., Otim-Nape, G.W., Legg, J.P., & Fargette, D. (1997). African cassava mosaic virus disease: the magnitude of the problem. *African Journal of Root and Tuber Crops*, 2(1-2), 13-19.

[12] Houngue, J.A., Zandjanakou-Tachin, M., Ngalle, H.B., Pita, J.S., Cacaï, G.H.T., Ngatat, S.E., ... & Ambang, Z. (2022). Evaluation of resistance to cassava mosaic disease in selected African cassava cultivars using combined molecular and phenotypic tools. *Physiological and Molecular Plant Pathology*, 116, 101709. DOI: 10.1016/j.pmpp.2021.101709

[13] APNI. (2020). *Training Agriculture Extension Service Providers*. African Plant Nutrition Institute.

[14] Akinola, A.A., Alene, A.D., Adeyemo, R., Sanogo, D., & Olanrewaju, A.S. (2019). Organisational capacities and management of agricultural extension services in Nigeria: Current status. *South African Journal of Agricultural Extension*, 47(2), 118-127.

[15] Ramcharan, A., Baranowski, K., McCloskey, P., Ahmed, B., Legg, J., & Hughes, D.P. (2017). Deep learning for image-based cassava disease detection. *Frontiers in Plant Science*, 8, 1852. DOI: 10.3389/fpls.2017.01852

[16] Singh, V., Sharma, N., & Singh, S. (2024). MiXceptionLeaf: A novel deep learning approach for cassava leaf disease classification. *Proceedings of CVIP 2024*.

[17] Zhang, K., Wu, Q., & Chen, Y. (2024). CDDNet: Cross-domain deep network for cassava disease detection. *Computers and Electronics in Agriculture*, 218, 108721.

[18] IITA. (2018). *African farmers get new help against cassava diseases: Nuru, their artificially intelligent assistant*. https://www.iita.org/news-item/african-farmers-get-new-help-against-cassava-diseases-nuru/

[19] Mrisho, L.M., Oerke, E.C., Hoogendoorn, J.,"; &"; Hughes, D.P. (2020). Accuracy of a smartphone-based object detection model, PlantVillage Nuru, in identifying the foliar symptoms of the viral diseases of cassava–CMD and CBSD. *Frontiers in Plant Science*, 11, 590889. DOI: 10.3389/fpls.2020.590889

[20] Ramcharan, A., McCloskey, P., Baranowski, K., Mbilinyi, N., Mrisho, L., Ndalahwa, M., ... & Hughes, D.P. (2019). A mobile-based deep learning model for cassava disease diagnosis. *Frontiers in Plant Science*, 10, 272. DOI: 10.3389/fpls.2019.00272

[21] Samek, W., Montavon, G., Vedaldi, A., Hansen, L.K., & Müller, K.R. (Eds.). (2019). *Explainable AI: interpreting, explaining and visualizing deep learning*. Springer Nature.

[22] Ahoya, D.K.D., Arouna, A., Diagne, A., & Mensah, A.C. (2024). Factors influencing adoption of the PlantVillage Nuru application for cassava mosaic disease diagnosis among farmers in Benin. *Agriculture*, 14(11), 2001. DOI: 10.3390/agriculture14112001

[23] Liu, H., Li, C., Wu, Q., & Lee, Y.J. (2024). Visual instruction tuning. *Advances in Neural Information Processing Systems*, 36.

[24] Sharma, P., Kumar, R., & Singh, A. (2024). LLaVA-PlantDiag: Expert-level visual question answering for plant disease diagnosis. *Proceedings of IEEE IJCNN 2024*. DOI: 10.1109/IJCNN60899.2024.10651096

[25] Wang, Y., Zhang, X., & Liu, J. (2024). Agri-LLaVA: A multimodal large language model for agricultural applications. *arXiv preprint arXiv:2412.02158*.

[26] Shinoda, R., Zhao, H., & Tanaka, K. (2025). AgroBench: Vision-language model benchmark in agriculture. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*.

[27] Chen, Y., Wang, X., & Li, Z. (2024). ITIMCA: Image-text information and cross-attention for multi-modal cassava leaf disease classification. *Crop Protection*, 180, 106618.

[28] Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. *International Conference on Learning Representations*. arXiv:2106.09685

[29] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. *Advances in Neural Information Processing Systems*, 36. arXiv:2305.14314

[30] Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., ... & Yang, J. (2025). Qwen2.5-VL technical report. *arXiv preprint arXiv:2502.13923*.