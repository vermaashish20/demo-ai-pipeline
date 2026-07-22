from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Paths configuration
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"

# Environment Dataset Paths
DEV_RAW_DATA_PATH = DATASET_DIR / "dev" / "raw" / "heart_disease_risk_2026.csv"
DEV_PROCESSED_DATA_PATH = DATASET_DIR / "dev" / "processed" / "ml_processed_data.csv"

MICRO_RAW_DATA_PATH = DATASET_DIR / "pytest-micro-data" / "raw" / "heart_disease_risk_2026.csv"
MICRO_PROCESSED_DATA_PATH = DATASET_DIR / "pytest-micro-data" / "processed" / "ml_processed_data.csv"

GOLDEN_RAW_DATA_PATH = DATASET_DIR / "golden-data" / "raw" / "heart_disease_risk_2026.csv"
GOLDEN_PROCESSED_DATA_PATH = DATASET_DIR / "golden-data" / "processed" / "ml_processed_data.csv"

# Default Model Configuration
DEFAULT_MODEL_NAME = "random_forest_classifier.joblib"
DEFAULT_PREPROCESSOR_NAME = "preprocessor.joblib"

# Server Settings
APP_NAME = "Heart Disease Risk Prediction API"
APP_VERSION = "1.0.0"
HOST = "0.0.0.0"
PORT = 8000

#==== WANDB Configs
# WandB Registry & Environment Settings
SCRIPT_ENV = "dev"  # Can be dev, ci, or prod
WANDB_ENTITY = "mahtech"  # team name of wandb 
WANDB_PROJECT = "mtech-demo-experiments"
WANDB_PREFIX = "wandb-registry"
# Dataset Registry and Collection
WANDB_REGISTRY_DATASET = f"{WANDB_PREFIX}-dataset"
WANDB_COLLECTION_GOLDEN_DATASET = "golden-dataset"
WANDB_COLLECTION_MICRO_DATASET = "micro-ci-pytest-dataset"
WANDB_COLLECTION_RAW_DATASET = "raw-dataset"
# Model Registry and collection
WANDB_REGISTRY_MODEL = f"{WANDB_PREFIX}-model"
WANDB_COLLECTION_MODEL = "heart-risk"
WANDB_COLLECTION_PROCESSOR = "heart-risk-processor"

