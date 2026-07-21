import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import wandb

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    WANDB_ENTITY, WANDB_PROJECT, WANDB_REGISTRY, 
    WANDB_GOLDEN_DATASET, WANDB_MICRO_DATASET
)

BASE_DIR = Path(__file__).resolve().parent.parent
TARGET_FEATURE = 'has_heart_disease'
# define features
NUMERICAL_FEATURE = [
    'age', 'resting_bp_systolic', 'resting_bp_diastolic', 'hdl', 'ldl',
    'triglycerides', 'resting_heart_rate', 'max_heart_rate_achieved',
    'exercise_minutes_per_week', 'daily_steps', 'hba1c', 'bmi',
    'st_depression', 'alcohol_units_per_week', 'sleep_hours',
    'stress_score', 'diet_quality_score'
]
CATEGORICAL_FEATURE = [
    'sex', 'chest_pain_type', 'smoker_status',
    'exercise_induced_angina', 'family_history', 'wearable_owner'
]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (true/false).")

def main():
    parser = argparse.ArgumentParser(
        description="Train ML Classification Model for Heart Disease Risk Prediction.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--env", "-e",
        type=str,
        default="dev",
        choices=["dev", "local", "ci-branch", "micro", "ci-prod", "golden"],
        help="Target environment (dev, ci-branch, ci-prod)"
    )
    parser.add_argument(
        "--input-dataset", "-i",
        type=str,
        default=None,
        help="Custom path to input CSV dataset"
    )
    parser.add_argument(
        "--output-path", "-o",
        type=str,
        default="models",
        help="Path or directory where trained model artifact will be saved"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="random_forest_classifier",
        choices=["random_forest_classifier", "random-forest-classifier", "rfc"],
        help="Model algorithm to train"
    )
    parser.add_argument(
        "--wandb", "-w",
        type=str2bool,
        default=False,
        help="Enable Weights & Biases (W&B) experiment logging (default: false)"
    )

    args = parser.parse_args()
    env_name = args.env.lower()

    # 1. Initialize W&B Run
    run_rf = None
    if args.wandb or env_name in ("ci-branch", "micro", "ci-prod", "golden"):
        try:
            RUN_NAME = f"random-forest-run-{env_name}"
            run_rf = wandb.init(
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                name=RUN_NAME,
                config={
                    "algorithm": "RandomForest",
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42,
                    "environment": env_name,
                },
            )
            print(f"W&B Experiment Tracking enabled for environment: '{env_name}'")
        except Exception as e:
            run_rf = None
            print(f"Notice: W&B initialization skipped ({e}). Continuing training...")

    # Dataset Loading
    input_path = None
    if args.input_dataset:
        input_path = Path(args.input_dataset)
    else:
        if env_name in ("ci-branch", "micro"):
            print("CI environment detected. Pulling micro dataset from WandB Registry...")
            artifact_path = f"{WANDB_ENTITY}/{WANDB_REGISTRY}/{WANDB_MICRO_DATASET}:production"
            if run_rf:
                artifact = run_rf.use_artifact(artifact_path)
            else:
                api = wandb.Api()
                artifact = api.artifact(artifact_path)
            download_dir = artifact.download()
            csv_files = list(Path(download_dir).glob("*.csv"))
            input_path = csv_files[0] if csv_files else None
        elif env_name in ("ci-prod", "golden"):
            print("Prod environment detected. Pulling golden dataset from WandB Registry...")
            artifact_path = f"{WANDB_ENTITY}/{WANDB_REGISTRY}/{WANDB_GOLDEN_DATASET}:production"
            if run_rf:
                artifact = run_rf.use_artifact(artifact_path)
            else:
                api = wandb.Api()
                artifact = api.artifact(artifact_path)
            download_dir = artifact.download()
            csv_files = list(Path(download_dir).glob("*.csv"))
            input_path = csv_files[0] if csv_files else None
        else:
            input_path = BASE_DIR / "dataset" / "dev" / "raw" / "heart_disease_risk_2026.csv"

    if input_path is None or not input_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Dataset loaded. Shape: {df.shape}")

    # Drop redundant columns
    cols_to_drop = ["cholesterol_total", "fasting_blood_sugar"]
    if "patient_id" in df.columns:
        cols_to_drop.append("patient_id")
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    print(f"Dataset shape after dropping redundant columns: {df.shape}")

    if TARGET_FEATURE not in df.columns:
        raise KeyError(f"Target column '{TARGET_FEATURE}' not found in dataset.")

    X = df[NUMERICAL_FEATURE + CATEGORICAL_FEATURE]
    y = df[TARGET_FEATURE]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        [
            ('num-scaler', StandardScaler(), NUMERICAL_FEATURE),
            ('cat-encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), CATEGORICAL_FEATURE)
        ]
    )

    rforest_classifier = Pipeline(
        steps=[
            ("data-preprocessor", preprocessor),
            (
                "rforest-classifier",
                RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42
                ),
            ),
        ]
    )

    print("Training RandomForestClassifier Pipeline...")
    rforestM = rforest_classifier.fit(X_train, y_train)

    y_pred = rforestM.predict(X_test)
    y_probas = rforestM.predict_proba(X_test)
    class_names = [str(cls) for cls in rforestM.classes_]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probas[:, 1])

    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC AUC: {auc:.4f}")

    # Save Models locally
    output_path_dir = BASE_DIR / args.output_path if not Path(args.output_path).is_absolute() else Path(args.output_path)
    output_path_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor_path = output_path_dir / "preprocessor.joblib"
    model_path = output_path_dir / "random_forest_classifier.joblib"

    # Extract the preprocessor from pipeline to save separately
    fitted_preprocessor = rforest_classifier.named_steps["data-preprocessor"]
    joblib.dump(fitted_preprocessor, preprocessor_path)
    joblib.dump(rforest_classifier, model_path)
    
    if run_rf is not None and wandb.run is not None:
        wandb.log({
            "accuracy": acc,
            "precision": prec,
            "recall": recall,
            "f1_score": f1,
            "roc_auc_score": auc,
        })

        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=np.asarray(y_test),
                preds=y_pred,
                class_names=class_names
            ),
            "roc_curve": wandb.plot.roc_curve(
                y_true=np.asarray(y_test),
                y_probas=y_probas,
                labels=class_names
            ),
            "precision_recall_curve": wandb.plot.pr_curve(
                y_true=np.asarray(y_test),
                y_probas=y_probas,
                labels=class_names
            ),
        })

        preprocessor_artifact = wandb.Artifact(
            name="heart-risk-preprocessor",
            type="model",
            description="ColumnTransformer for encoding and scaling tabular features",
        )
        preprocessor_artifact.add_file(str(preprocessor_path))
        logged_prep = run_rf.log_artifact(preprocessor_artifact)

        artifact = wandb.Artifact(
            name="heart-risk-ml",
            type="model",
            description="Random Forest pipeline model with preprocessor",
        )
        artifact.add_file(str(model_path))
        logged_artifact = run_rf.log_artifact(artifact)

        if env_name in ("ci-prod", "golden"):
            run_rf.link_artifact(
                artifact=logged_prep,
                target_path="wandb-registry-model/heart-risk-processor",
                aliases=["production"],
            )
            run_rf.link_artifact(
                artifact=logged_artifact,
                target_path="wandb-registry-model/heart-risk",
                aliases=["production"],
            )

        run_rf.finish()

if __name__ == "__main__":
    main()
