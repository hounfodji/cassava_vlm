#!/bin/bash
# Download PlantVillage dataset from Kaggle
# Requires Kaggle API credentials (~/.kaggle/kaggle.json)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== PlantVillage Dataset Downloader ===${NC}"

# Check if kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo -e "${RED}✗ Kaggle CLI not found${NC}"
    echo "Install with: pip install kaggle"
    exit 1
fi

# Check for Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo -e "${YELLOW}⚠ Kaggle credentials not found${NC}"
    echo "Please set up Kaggle API credentials:"
    echo "1. Go to https://www.kaggle.com/settings/account"
    echo "2. Click 'Create New API Token'"
    echo "3. Save kaggle.json to ~/.kaggle/"
    echo "4. chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# Create data directory
DATA_DIR="data/raw/plantvillage"
mkdir -p "$DATA_DIR"

echo -e "\n${GREEN}Downloading PlantVillage dataset...${NC}"
echo "This may take several minutes (dataset is ~2GB)"

# Download dataset
cd "$DATA_DIR"
kaggle datasets download -d emmarex/plantdisease

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Download complete${NC}"
else
    echo -e "${RED}✗ Download failed${NC}"
    exit 1
fi

# Extract dataset
echo -e "\n${GREEN}Extracting dataset...${NC}"
unzip -q plantdisease.zip

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Extraction complete${NC}"
else
    echo -e "${RED}✗ Extraction failed${NC}"
    exit 1
fi

# Clean up zip file
rm plantdisease.zip

# Count images
echo -e "\n${GREEN}Verifying dataset...${NC}"
NUM_IMAGES=$(find . -name "*.jpg" -o -name "*.JPG" -o -name "*.png" | wc -l)
NUM_CLASSES=$(find . -maxdepth 1 -type d | wc -l)

echo "Found $NUM_IMAGES images in $((NUM_CLASSES - 1)) classes"

cd - > /dev/null  # Return to original directory

echo -e "\n${GREEN}=== Download Complete ===${NC}"
echo "PlantVillage dataset is ready at: $DATA_DIR"
echo ""
echo "Next steps:"
echo "1. Create VQA pairs: python data_preparation/create_plantvillage_vqa.py"
echo "2. Validate dataset: python data_preparation/validate_datasets.py data/stage1/plantvillage_vqa.json"
