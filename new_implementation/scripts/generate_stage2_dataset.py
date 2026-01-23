#!/usr/bin/env python3
"""
Script de génération du Dataset Stage 2 - VQA Conversationnel
Approche hybride: Templates (80%) + gpt-oss via Ollama (20%)

Usage:
    python generate_stage2_dataset.py --test --n_samples 100
    python generate_stage2_dataset.py --output ../data/stage2/stage2_vqa_dataset.json
    python generate_stage2_dataset.py --stats
"""

import json
import random
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
import requests

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
STAGE1_DATA = DATA_DIR / "stage1" / "cassava_train_balanced.json"
KB_FILE = BASE_DIR.parent / "redaction" / "base_connaissance_maladies_manioc.md"
OUTPUT_DIR = DATA_DIR / "stage2"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gpt-oss:latest"

# Ratio templates vs LLM
TEMPLATE_RATIO = 0.80  # 80% templates, 20% LLM

# Mapping source -> (classe, sévérité)
SOURCE_MAPPING = {
    "cbsd-1": ("CBSD", "severite_1"),
    "cbsd-2": ("CBSD", "severite_2"),
    "cmd_001": ("CMD", "severite_1"),
    "cmd_002": ("CMD", "severite_2"),
    "cmd_003": ("CMD", "severite_3"),
    "cmd_004": ("CMD", "severite_4"),
    "cmd_005": ("CMD", "severite_5"),
    "healthy_001": ("Healthy", "general"),
    "healthy_002": ("Healthy", "general"),
    "kaggle": None,  # Utilise le label
    "cassava_train_cmd": ("CMD", "general"),
    "cassava_train_cbsd": ("CBSD", "general"),
    "cassava_train_cgm": ("CGM", "general"),
    "cassava_train_cbb": ("CBB", "general"),
    "cassava_train_healthy": ("Healthy", "general"),
}

# Mapping label -> classe (pour source kaggle)
LABEL_TO_CLASS = {
    0: "CBB",
    1: "CBSD",
    2: "CGM",
    3: "CMD",
    4: "Healthy"
}

# ============================================================
# BASE DE CONNAISSANCES STRUCTURÉE
# ============================================================

KNOWLEDGE_BASE = {
    "CMD": {
        "nom_complet": "Mosaïque du manioc",
        "nom_scientifique": "Cassava Mosaic Disease",
        "agent": "Begomovirus (ACMV, EACMV)",
        "vecteur": "Mouche blanche (Bemisia tabaci)",
        "severites": {
            "severite_1": {
                "description": "Très légère/Précoce",
                "surface_affectee": "<5%",
                "symptomes": [
                    "Chlorose très légère à peine perceptible",
                    "Quelques taches jaune pâle isolées sur fond vert normal",
                    "Pas de déformation des feuilles"
                ],
                "impact_rendement": "0-5%",
                "action": "Surveiller l'évolution, marquer les plants pour suivi"
            },
            "severite_2": {
                "description": "Légère",
                "surface_affectee": "5-15%",
                "symptomes": [
                    "Chlorose légère avec motif mosaïque faible",
                    "Motifs jaune-vert pâle sur fond vert normal",
                    "Aucune déformation ni réduction de taille des folioles",
                    "Croissance normale ou légèrement réduite"
                ],
                "impact_rendement": "5-15%",
                "action": "Épuration si peu de plants affectés, sinon surveillance"
            },
            "severite_3": {
                "description": "Modérée",
                "surface_affectee": "15-30%",
                "symptomes": [
                    "Mosaïque prononcée visible sur l'ensemble de la feuille",
                    "Motifs jaune-verdâtre bien marqués, parfois presque blancs",
                    "Rétrécissement modéré du tiers inférieur des folioles",
                    "Légère torsion foliaire",
                    "Chlorose modérée des pousses terminales"
                ],
                "impact_rendement": "20-40%",
                "action": "Arracher et brûler les plants infectés, ne jamais utiliser comme boutures"
            },
            "severite_4": {
                "description": "Sévère",
                "surface_affectee": "30-66%",
                "symptomes": [
                    "Mosaïque sévère avec déformation prononcée",
                    "Chlorose sévère, zones jaune vif à blanchâtres dominantes",
                    "Déformation des deux tiers des folioles",
                    "Réduction générale de la taille des feuilles",
                    "Début de rabougrissement visible des pousses"
                ],
                "impact_rendement": "50-70%",
                "action": "Épuration urgente, détruire par brûlage, replanter variété résistante"
            },
            "severite_5": {
                "description": "Très sévère/Critique",
                "surface_affectee": ">80%",
                "symptomes": [
                    "Symptômes extrêmes généralisés",
                    "Chlorose très sévère, feuilles majoritairement jaune-blanc",
                    "Feuilles très déformées, tordues, malformées (forme chandelier)",
                    "Folioles filiformes dans les cas extrêmes",
                    "Rabougrissement sévère de la plante entière",
                    "Tubercules très réduits ou absents"
                ],
                "impact_rendement": "70-100%",
                "action": "Destruction immédiate, traitement urgent de toute la parcelle"
            },
            "general": {
                "description": "Général (sévérité non spécifiée)",
                "symptomes": [
                    "Motifs mosaïque jaune-vert sur les feuilles",
                    "Déformation possible des folioles",
                    "Nanisme dans les cas avancés"
                ],
                "impact_rendement": "Variable selon sévérité",
                "action": "Évaluer la sévérité et agir en conséquence"
            }
        },
        "varietes_resistantes": ["TME 419", "TMS 30572", "TMS 98/0505", "NAROCASS 1"],
        "prevention": [
            "Utiliser des boutures de sources certifiées",
            "Inspecter visuellement avant plantation",
            "Couper 10-15 cm au-dessus du sol",
            "Épuration des plants malades",
            "Planter en début de saison des pluies"
        ]
    },
    "CBSD": {
        "nom_complet": "Striure brune du manioc",
        "nom_scientifique": "Cassava Brown Streak Disease",
        "agent": "Ipomovirus (CBSV, UCBSV)",
        "vecteur": "Mouche blanche (transmission semi-persistante)",
        "severites": {
            "severite_1": {
                "description": "Légère/Précoce",
                "symptomes": [
                    "Chlorose veinaire - jaunissement le long des veines secondaires",
                    "Motifs 'en plume' partant des nervures vers les bords",
                    "Taches chlorotiques dispersées sur le limbe",
                    "Affecte les FEUILLES MATURES (partie inférieure), PAS les jeunes",
                    "Tiges généralement sans symptômes visibles",
                    "Racines souvent sans symptômes (infection latente)"
                ],
                "impact_rendement": "18-25%",
                "action": "Surveillance accrue, vérifier racines par sondage, éviter comme boutures"
            },
            "severite_2": {
                "description": "Modérée à Avancée",
                "symptomes": [
                    "Coalescence des chloroses en grandes plages jaunes",
                    "Jaunissement généralisé du limbe",
                    "Nécroses foliaires (zones brunes mortes)",
                    "Abscission prématurée des feuilles inférieures",
                    "Stries brunes caractéristiques sur tiges vertes",
                    "Lésions allongées brun foncé comme des griffures",
                    "Nécrose racinaire interne jaune-brun à brun-noir",
                    "Texture liégeuse/sèche des racines"
                ],
                "impact_rendement": "50-100%",
                "action": "Destruction immédiate, récolte précoce plants voisins, replanter variétés tolérantes"
            },
            "general": {
                "description": "Général",
                "symptomes": [
                    "Chlorose sur feuilles matures (pas les jeunes)",
                    "Stries brunes possibles sur tiges",
                    "Nécrose brune interne des racines"
                ],
                "impact_rendement": "18-100% selon sévérité",
                "action": "Vérifier les racines, évaluer sévérité"
            }
        },
        "alerte": "CBSD non encore présent en Afrique de l'Ouest - risque d'introduction élevé",
        "varietes_tolerantes": ["Kiroba", "Namikonga", "NAROCASS 1", "NASE 19"],
        "prevention": [
            "Boutures saines certifiées (tests RT-PCR)",
            "Arracher et brûler immédiatement plants symptomatiques",
            "Récolter quelques plants avant récolte principale",
            "Récolte précoce (9-12 mois au lieu de 18-24)",
            "Quarantaine - pas de boutures des zones infectées"
        ]
    },
    "CGM": {
        "nom_complet": "Acarien vert du manioc",
        "nom_scientifique": "Cassava Green Mite (Mononychellus tanajoa)",
        "agent": "Acarien (Mononychellus tanajoa)",
        "severites": {
            "general": {
                "description": "Dégâts d'acariens",
                "symptomes": [
                    "Minuscules taches jaunes chlorotiques (piqûres d'épingle)",
                    "Aspect moucheté sur face supérieure des feuilles",
                    "Extension de la chlorose (bronzage)",
                    "Feuilles deviennent mouchetées puis jaunissent",
                    "Symptôme 'Chandelle' - feuilles terminales meurent",
                    "Apex prend aspect de chandelle/bougie",
                    "Folioles petites, non développées, étroites"
                ],
                "impact_rendement": "13-80%",
                "action": "Favoriser prédateurs naturels, planter tôt, irrigation si possible"
            }
        },
        "conditions_favorables": [
            "Saison sèche (absence de pluie)",
            "Température 28-32°C",
            "Faible humidité",
            "Stress hydrique"
        ],
        "lutte_biologique": "Typhlodromalus aripo (réduction jusqu'à 90%)",
        "prevention": [
            "Planter tôt en saison des pluies",
            "Irrigation par aspersion si possible",
            "Paillage pour maintenir humidité",
            "Association avec pois d'Angole",
            "NE PAS utiliser de pesticides chimiques"
        ]
    },
    "CBB": {
        "nom_complet": "Bactériose du manioc",
        "nom_scientifique": "Cassava Bacterial Blight (Xanthomonas)",
        "agent": "Xanthomonas phaseoli pv. manihotis",
        "severites": {
            "general": {
                "description": "Infection bactérienne",
                "symptomes": [
                    "Taches angulaires aqueuses (aspect mouillé/huileux)",
                    "Couleur vert foncé, limitées par les nervures",
                    "Exsudation gommeuse - gouttelettes blanches crémeuses",
                    "Brunissement progressif des taches",
                    "Halo chlorotique jaune autour des lésions",
                    "Brûlure marginale (leaf scorch)",
                    "Flétrissement foliaire, enroulement vers le haut",
                    "Exsudat gommeux crème à jaune sur écorce",
                    "Défoliation sévère possible"
                ],
                "impact_rendement": "12-100%",
                "action": "Élimination en période sèche, désinfection outils, rotation culturale"
            }
        },
        "test_diagnostic": "Bacterial Streaming - couper tissu dans eau, observer flux bactérien (fumée blanche)",
        "conditions_favorables": [
            "Humidité 90-100%",
            "Température 22-30°C",
            "Saison des pluies",
            "Présence de blessures"
        ],
        "prevention": [
            "Boutures de sources certifiées",
            "Trempage boutures solution cuivre ou eau chaude (49°C, 5h)",
            "Désinfection outils (flamme ou eau de Javel)",
            "Rotation culturale 1-2 saisons sans manioc",
            "Apport de potassium si sols carencés"
        ]
    },
    "Healthy": {
        "nom_complet": "Plant sain",
        "severites": {
            "general": {
                "description": "Plant en bonne santé",
                "caracteristiques": [
                    "Couleur vert foncé à vert moyen uniforme",
                    "Surface brillante/cireuse",
                    "Jeunes feuilles parfois teintées violet/pourpre",
                    "Forme lancéolée/palmée avec 5-7 lobes",
                    "Folioles symétriques par rapport à nervure centrale",
                    "Bords lisses et entiers",
                    "Surface lisse et glabre",
                    "Consistance souple mais ferme",
                    "Tiges sans lésions ni exsudat",
                    "Racines chair blanche/crème uniforme, texture ferme"
                ]
            }
        },
        "variations_normales": [
            "Pétioles verts, rouges, pourpres selon variété",
            "Jeunes feuilles plus claires ou pourprées",
            "Jaunissement progressif des vieilles feuilles (normal)",
            "Flétrissement réversible après stress hydrique"
        ]
    }
}

# ============================================================
# TEMPLATES DE QUESTIONS/RÉPONSES
# ============================================================

QUESTION_TEMPLATES = {
    "identification": [
        "Quelle maladie affecte cette feuille de manioc ?",
        "Qu'est-ce qui ne va pas avec mon manioc ?",
        "Pouvez-vous identifier la maladie sur cette image ?",
        "Mon manioc est-il malade ? Si oui, quelle maladie ?",
        "Quel est le problème avec cette plante de manioc ?",
        "C'est quoi ces taches sur mes feuilles de manioc ?",
        "Ma plante de manioc a un problème, qu'est-ce que c'est ?",
        "Identifiez la maladie présente sur cette feuille.",
        "Cette feuille de manioc est-elle saine ou malade ?",
        "Diagnostic de cette feuille de manioc s'il vous plaît."
    ],
    "descriptif": [
        "Décrivez les symptômes visibles sur cette feuille.",
        "Quels sont les signes de maladie que vous observez ?",
        "Pouvez-vous décrire l'état de cette plante ?",
        "Que voyez-vous sur cette image de manioc ?",
        "Décrivez ce que vous observez sur cette feuille.",
        "Quelles sont les anomalies visibles ?",
        "Expliquez les symptômes présents sur cette image."
    ],
    "recommandation": [
        "Que dois-je faire pour traiter cette maladie ?",
        "Comment puis-je sauver mon manioc ?",
        "Quelles actions recommandez-vous ?",
        "Comment traiter ce problème ?",
        "Que faire pour éviter que ça se propage ?",
        "Quels traitements sont possibles ?",
        "Comment protéger mes autres plants ?",
        "Donnez-moi des conseils pour gérer ce problème.",
        "Quelles mesures prendre immédiatement ?"
    ],
    "severite": [
        "C'est grave comme maladie ?",
        "Quel est le niveau de sévérité ?",
        "Puis-je encore sauver ma récolte ?",
        "À quel stade en est la maladie ?",
        "Est-ce que c'est très avancé ?",
        "Quelle est la gravité de l'infection ?",
        "Mon rendement sera-t-il affecté ?"
    ],
    "comparaison": [
        "Est-ce la mosaïque ou la striure brune ?",
        "Comment différencier cette maladie des autres ?",
        "Est-ce une maladie ou une carence nutritionnelle ?",
        "Comment savoir si c'est CMD ou CBSD ?",
        "Est-ce un problème de maladie ou d'insectes ?"
    ],
    "raisonnement": [
        "Pourquoi mes feuilles jaunissent-elles ?",
        "D'où vient cette maladie ?",
        "Comment la maladie s'est-elle propagée ?",
        "Qu'est-ce qui cause ces symptômes ?",
        "Pourquoi les feuilles sont-elles déformées ?"
    ]
}

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def get_class_and_severity(sample: Dict) -> Tuple[str, str]:
    """Détermine la classe et la sévérité d'un échantillon."""
    metadata = sample.get("metadata", {})
    source = metadata.get("source", "kaggle")
    label = metadata.get("label", 4)

    if source in SOURCE_MAPPING and SOURCE_MAPPING[source] is not None:
        return SOURCE_MAPPING[source]

    # Source kaggle ou inconnue - utiliser le label
    class_name = LABEL_TO_CLASS.get(label, "Healthy")
    return (class_name, "general")


def get_image_path(sample: Dict) -> str:
    """Extrait le chemin de l'image d'un échantillon."""
    conversations = sample.get("conversations", [])
    for turn in conversations:
        if "<img>" in turn.get("value", ""):
            match = re.search(r'<img>(.*?)</img>', turn["value"])
            if match:
                return match.group(1)
    return ""


def call_ollama(prompt: str, max_tokens: int = 300) -> str:
    """Appelle Ollama pour générer du texte."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            },
            timeout=120  # Augmenté pour gpt-oss
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        return ""
    except Exception as e:
        print(f"Erreur Ollama: {e}")
        return ""


def generate_template_response(disease_class: str, severity: str, qa_type: str) -> Tuple[str, str]:
    """Génère une paire Q/A basée sur les templates."""
    kb = KNOWLEDGE_BASE.get(disease_class, KNOWLEDGE_BASE["Healthy"])
    sev_info = kb.get("severites", {}).get(severity, kb.get("severites", {}).get("general", {}))

    # Sélectionner une question aléatoire
    questions = QUESTION_TEMPLATES.get(qa_type, QUESTION_TEMPLATES["identification"])
    question = random.choice(questions)

    # Construire la réponse
    if disease_class == "Healthy":
        response = generate_healthy_response(kb, sev_info)
    else:
        response = generate_disease_response(disease_class, kb, sev_info, qa_type)

    return question, response


def generate_healthy_response(kb: Dict, sev_info: Dict) -> str:
    """Génère une réponse pour un plant sain."""
    caracteristiques = sev_info.get("caracteristiques", [])
    carac_text = random.sample(caracteristiques, min(3, len(caracteristiques)))

    templates = [
        f"Cette feuille de manioc est **saine**. Elle présente les caractéristiques normales d'un plant en bonne santé : {', '.join(carac_text)}. Continuez à bien entretenir votre culture avec un arrosage régulier et une fertilisation adaptée.",
        f"Bonne nouvelle ! Votre manioc est en **bonne santé**. Je n'observe aucun signe de maladie. Les caractéristiques visibles ({', '.join(carac_text[:2])}) sont tout à fait normales pour un plant sain.",
        f"Ce plant de manioc ne présente **aucune maladie**. Les feuilles montrent une coloration et une forme normales. Veillez à maintenir de bonnes pratiques culturales pour préserver cette santé."
    ]
    return random.choice(templates)


def generate_disease_response(disease_class: str, kb: Dict, sev_info: Dict, qa_type: str) -> str:
    """Génère une réponse pour une maladie."""
    nom_complet = kb.get("nom_complet", disease_class)
    description = sev_info.get("description", "")
    symptomes = sev_info.get("symptomes", [])
    impact = sev_info.get("impact_rendement", "Variable")
    action = sev_info.get("action", "Consulter un expert")
    prevention = kb.get("prevention", [])

    symptomes_text = ", ".join(random.sample(symptomes, min(3, len(symptomes)))) if symptomes else "symptômes caractéristiques"

    if qa_type == "identification":
        templates = [
            f"Cette feuille présente des symptômes de **{nom_complet} ({disease_class})**. Niveau de sévérité : {description}. Les signes visibles incluent : {symptomes_text}.",
            f"D'après l'image, votre manioc est atteint de la **{nom_complet}** ({disease_class}). J'observe : {symptomes_text}. C'est un stade {description.lower()}.",
            f"Diagnostic : **{disease_class}** - {nom_complet}. Sévérité : {description}. Symptômes observés : {symptomes_text}."
        ]
    elif qa_type == "descriptif":
        templates = [
            f"J'observe les symptômes suivants caractéristiques de {nom_complet} ({disease_class}) : {symptomes_text}. Le stade est {description.lower()}.",
            f"Cette image montre : {symptomes_text}. Ces signes sont typiques de la {nom_complet} au stade {description.lower()}."
        ]
    elif qa_type == "recommandation":
        prevention_text = ". ".join(random.sample(prevention, min(3, len(prevention)))) if prevention else action
        templates = [
            f"Pour cette {nom_complet} ({disease_class}) au stade {description.lower()}, je recommande : {action}. Mesures préventives : {prevention_text}.",
            f"Actions à prendre pour la {disease_class} : {action}. Pour protéger vos autres plants : {prevention_text}."
        ]
    elif qa_type == "severite":
        templates = [
            f"Cette {nom_complet} est au stade **{description}**. Impact sur le rendement : **{impact}** de pertes estimées. {action}.",
            f"Niveau de sévérité : **{description}**. Avec une perte de rendement de {impact}, il est important d'agir : {action}."
        ]
    elif qa_type == "comparaison":
        agent = kb.get("agent", "pathogène")
        templates = [
            f"Il s'agit de **{nom_complet} ({disease_class})**, causée par {agent}. Les symptômes distinctifs sont : {symptomes_text}.",
            f"Cette maladie est la {disease_class}, différente des autres par : {symptomes_text}. Agent causal : {agent}."
        ]
    else:  # raisonnement
        agent = kb.get("agent", "pathogène")
        vecteur = kb.get("vecteur", "transmission directe")
        templates = [
            f"Ces symptômes sont causés par {agent}, responsable de la {nom_complet}. La transmission se fait par {vecteur}.",
            f"La cause de ces problèmes est la {disease_class}, due à {agent}. Mode de propagation : {vecteur}."
        ]

    return random.choice(templates)


def generate_llm_variation(disease_class: str, severity: str, base_response: str) -> Tuple[str, str]:
    """Utilise Ollama pour générer une variation naturelle."""
    kb = KNOWLEDGE_BASE.get(disease_class, {})
    nom_complet = kb.get("nom_complet", disease_class)

    prompt = f"""Tu es un expert en agriculture africaine, spécialisé dans les maladies du manioc.
Un agriculteur te montre une image de son manioc atteint de {nom_complet} ({disease_class}).

Génère une question naturelle qu'un agriculteur francophone pourrait poser, suivie d'une réponse informative et accessible.
Utilise un langage simple et pratique.

Format de réponse (JSON):
{{"question": "...", "reponse": "..."}}

Ne génère que le JSON, rien d'autre."""

    response = call_ollama(prompt)

    try:
        # Parser la réponse JSON
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("question", ""), data.get("reponse", "")
    except:
        pass

    # Fallback sur template si LLM échoue
    return generate_template_response(disease_class, severity, "identification")


def generate_qa_pair(sample: Dict, use_llm: bool = False) -> Dict:
    """Génère une paire Q/A pour un échantillon."""
    disease_class, severity = get_class_and_severity(sample)
    image_path = get_image_path(sample)

    # Sélectionner le type de Q/A
    qa_types = ["identification"] * 25 + ["descriptif"] * 20 + ["recommandation"] * 20 + \
               ["raisonnement"] * 15 + ["severite"] * 10 + ["comparaison"] * 10
    qa_type = random.choice(qa_types)

    if use_llm:
        question, response = generate_llm_variation(disease_class, severity, "")
        method = "llm"
    else:
        question, response = generate_template_response(disease_class, severity, qa_type)
        method = "template"

    # Si la génération a échoué, utiliser un template de base
    if not question or not response:
        question, response = generate_template_response(disease_class, severity, qa_type)
        method = "template_fallback"

    return {
        "id": f"stage2_{disease_class.lower()}_{severity}_{sample.get('id', 'unknown')}",
        "conversations": [
            {
                "from": "user",
                "value": f"Picture 1: <img>{image_path}</img>\n{question}"
            },
            {
                "from": "assistant",
                "value": response
            }
        ],
        "metadata": {
            "source": sample.get("metadata", {}).get("source", "unknown"),
            "class": disease_class,
            "severity": severity,
            "qa_type": qa_type,
            "generation_method": method,
            "original_id": sample.get("id", "unknown")
        }
    }


def generate_dataset(samples: List[Dict], template_ratio: float = 0.8, verbose: bool = True) -> List[Dict]:
    """Génère le dataset Stage 2 complet."""
    dataset = []

    iterator = tqdm(samples, desc="Génération Q/A") if verbose else samples

    for sample in iterator:
        use_llm = random.random() > template_ratio
        qa_pair = generate_qa_pair(sample, use_llm=use_llm)
        dataset.append(qa_pair)

    return dataset


def compute_stats(dataset: List[Dict]) -> Dict:
    """Calcule les statistiques du dataset."""
    stats = {
        "total": len(dataset),
        "by_class": defaultdict(int),
        "by_severity": defaultdict(int),
        "by_qa_type": defaultdict(int),
        "by_method": defaultdict(int)
    }

    for item in dataset:
        meta = item.get("metadata", {})
        stats["by_class"][meta.get("class", "unknown")] += 1
        stats["by_severity"][meta.get("severity", "unknown")] += 1
        stats["by_qa_type"][meta.get("qa_type", "unknown")] += 1
        stats["by_method"][meta.get("generation_method", "unknown")] += 1

    return dict(stats)


def print_stats(stats: Dict):
    """Affiche les statistiques."""
    print("\n" + "=" * 60)
    print("STATISTIQUES DU DATASET STAGE 2")
    print("=" * 60)
    print(f"\nTotal: {stats['total']} paires Q/A")

    print("\nPar classe:")
    for cls, count in sorted(stats["by_class"].items()):
        pct = count / stats["total"] * 100
        print(f"  {cls}: {count} ({pct:.1f}%)")

    print("\nPar sévérité:")
    for sev, count in sorted(stats["by_severity"].items()):
        pct = count / stats["total"] * 100
        print(f"  {sev}: {count} ({pct:.1f}%)")

    print("\nPar type de Q/A:")
    for qa_type, count in sorted(stats["by_qa_type"].items()):
        pct = count / stats["total"] * 100
        print(f"  {qa_type}: {count} ({pct:.1f}%)")

    print("\nPar méthode de génération:")
    for method, count in sorted(stats["by_method"].items()):
        pct = count / stats["total"] * 100
        print(f"  {method}: {count} ({pct:.1f}%)")

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Génération Dataset Stage 2")
    parser.add_argument("--test", action="store_true", help="Mode test (100 samples)")
    parser.add_argument("--n_samples", type=int, default=100, help="Nombre de samples en mode test")
    parser.add_argument("--output", type=str, help="Fichier de sortie JSON")
    parser.add_argument("--stats", action="store_true", help="Afficher stats d'un dataset existant")
    parser.add_argument("--template_ratio", type=float, default=0.8, help="Ratio templates vs LLM")
    args = parser.parse_args()

    # Mode stats
    if args.stats and args.output:
        with open(args.output, 'r') as f:
            dataset = json.load(f)
        stats = compute_stats(dataset)
        print_stats(stats)
        return

    # Charger les données Stage 1
    print(f"Chargement des données depuis {STAGE1_DATA}...")
    with open(STAGE1_DATA, 'r') as f:
        stage1_data = json.load(f)
    print(f"  {len(stage1_data)} échantillons chargés")

    # Mode test
    if args.test:
        samples = random.sample(stage1_data, min(args.n_samples, len(stage1_data)))
        print(f"\nMode TEST: génération de {len(samples)} échantillons")
    else:
        samples = stage1_data
        print(f"\nMode COMPLET: génération de {len(samples)} échantillons")

    # Tester connexion Ollama
    print("\nTest connexion Ollama...")
    test_response = call_ollama("Réponds juste 'OK'", max_tokens=50)
    if test_response:
        print(f"  ✅ Ollama connecté ({OLLAMA_MODEL})")
    else:
        print(f"  ⚠️ Ollama non disponible - utilisation templates uniquement")
        args.template_ratio = 1.0

    # Générer le dataset
    print(f"\nGénération (ratio templates: {args.template_ratio * 100}%, LLM: {(1-args.template_ratio) * 100}%)...")
    dataset = generate_dataset(samples, template_ratio=args.template_ratio)

    # Statistiques
    stats = compute_stats(dataset)
    print_stats(stats)

    # Sauvegarder
    output_path = args.output or (OUTPUT_DIR / "stage2_vqa_test.json" if args.test else OUTPUT_DIR / "stage2_vqa_dataset.json")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Dataset sauvegardé: {output_path}")
    print(f"   Taille: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
