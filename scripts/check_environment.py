#!/usr/bin/env python3
"""
Script de vérification de l'environnement pour CassavaVLM
Vérifie que PyTorch, CUDA et les dépendances principales sont correctement installés
"""

import sys

def check_python_version():
    """Vérifie la version de Python"""
    print("=" * 60)
    print("1. VERSION PYTHON")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print()

def check_pytorch():
    """Vérifie l'installation de PyTorch et la détection GPU"""
    print("=" * 60)
    print("2. PYTORCH ET GPU")
    print("=" * 60)
    try:
        import torch
        print(f"✓ PyTorch installé: {torch.__version__}")
        print(f"✓ CUDA disponible: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"✓ Version CUDA: {torch.version.cuda}")
            print(f"✓ Nombre de GPUs: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                print(f"\n  GPU {i}:")
                print(f"    Nom: {torch.cuda.get_device_name(i)}")
                print(f"    Mémoire totale: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
                print(f"    Compute Capability: {torch.cuda.get_device_capability(i)}")

            # Test simple sur GPU
            print("\n  Test d'allocation mémoire GPU...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print(f"  ✓ Test réussi: multiplication matricielle sur GPU")
            del x, y, z
            torch.cuda.empty_cache()
        else:
            print("✗ CUDA non disponible - GPU ne sera pas utilisé")
    except ImportError:
        print("✗ PyTorch non installé")
        return False
    print()
    return True

def check_transformers():
    """Vérifie l'installation de transformers"""
    print("=" * 60)
    print("3. TRANSFORMERS (HUGGING FACE)")
    print("=" * 60)
    try:
        import transformers
        print(f"✓ Transformers installé: {transformers.__version__}")
    except ImportError:
        print("✗ Transformers non installé")
        return False
    print()
    return True

def check_vision_libs():
    """Vérifie les bibliothèques de vision"""
    print("=" * 60)
    print("4. BIBLIOTHÈQUES DE VISION")
    print("=" * 60)

    # PIL/Pillow
    try:
        from PIL import Image
        print(f"✓ Pillow installé")
    except ImportError:
        print("✗ Pillow non installé")

    # OpenCV
    try:
        import cv2
        print(f"✓ OpenCV installé: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV non installé")

    # torchvision
    try:
        import torchvision
        print(f"✓ torchvision installé: {torchvision.__version__}")
    except ImportError:
        print("✗ torchvision non installé")

    print()

def check_data_libs():
    """Vérifie les bibliothèques de manipulation de données"""
    print("=" * 60)
    print("5. BIBLIOTHÈQUES DE DONNÉES")
    print("=" * 60)

    # Pandas
    try:
        import pandas as pd
        print(f"✓ Pandas installé: {pd.__version__}")
    except ImportError:
        print("✗ Pandas non installé")

    # Numpy
    try:
        import numpy as np
        print(f"✓ NumPy installé: {np.__version__}")
    except ImportError:
        print("✗ NumPy non installé")

    # Matplotlib
    try:
        import matplotlib
        print(f"✓ Matplotlib installé: {matplotlib.__version__}")
    except ImportError:
        print("✗ Matplotlib non installé")

    # Seaborn
    try:
        import seaborn as sns
        print(f"✓ Seaborn installé: {sns.__version__}")
    except ImportError:
        print("✗ Seaborn non installé")

    print()

def check_qwen_utils():
    """Vérifie qwen-vl-utils"""
    print("=" * 60)
    print("6. QWEN VL UTILS")
    print("=" * 60)
    try:
        import qwen_vl_utils
        print(f"✓ qwen-vl-utils installé")
    except ImportError:
        print("✗ qwen-vl-utils non installé")
    print()

def main():
    """Fonction principale"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "VÉRIFICATION ENVIRONNEMENT CASSAVVLM" + " " * 11 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    check_python_version()
    pytorch_ok = check_pytorch()
    transformers_ok = check_transformers()
    check_vision_libs()
    check_data_libs()
    check_qwen_utils()

    print("=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    if pytorch_ok and transformers_ok:
        print("✓ Environnement prêt pour CassavaVLM!")
        print("  Vous pouvez passer à l'étape suivante: test du modèle Qwen2.5-VL")
    else:
        print("✗ Certaines dépendances manquent")
        print("  Veuillez installer les packages manquants avec:")
        print("  pip install -r requirements.txt")
    print("=" * 60)
    print()

if __name__ == "__main__":
    main()
