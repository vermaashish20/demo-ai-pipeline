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

# WandB Registry & Environment Settings
SCRIPT_ENV = "dev"  # Can be dev, ci, or prod
WANDB_ENTITY = "aithreads24"
WANDB_PROJECT = "Threads"
WANDB_REGISTRY = "wandb-registry-dataset"
WANDB_GOLDEN_DATASET = "golden-dataset"
WANDB_MICRO_DATASET = "micro-ci-pytest-dataset"
WANDB_RAW_DATASET = "raw-dataset"

