import os
from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

def load_test_data():
    """
    Load the saved test dataset.
    """
    root = Path.cwd()
    test_path = root / "data" / "test_data.csv"

    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at: {test_path}")

    df = pd.read_csv(test_path)

    X_test = df.drop(columns=["diagnosis"])
    y_test = df["diagnosis"]

    return X_test, y_test


def load_models():
    """
    Load all saved models from saved_models directory.
    """
    root = Path.cwd()
    model_dir = root / "saved_models"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found at: {model_dir}")

    models = {}

    for file in model_dir.glob("*.pkl"):
        model_name = file.stem
        models[model_name] = joblib.load(file)

    if not models:
        raise ValueError("No models found in saved_models directory.")

    return models


def evaluate_models():
    """
    Evaluate all saved models on test data
    and return comparison DataFrame.
    """
    X_test, y_test = load_test_data()
    models = load_models()

    results = []

    for model_name, model in models.items():
        y_pred = model.predict(X_test)

        # AUC requires probability scores
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        else:
            auc = None

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        results.append([
            model_name,
            acc,
            auc,
            precision,
            recall,
            f1,
            mcc
        ])

    results_df = pd.DataFrame(
        results,
        columns=["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    )

    return results_df


def main():
    print("🔎 Evaluating models...\n")

    results_df = evaluate_models()

    print("✅ Evaluation Completed.\n")
    print(results_df)

    # Save metrics file for README and Streamlit
    root = Path.cwd()
    results_df.to_csv(root / "model_metrics.csv", index=False)
    print("\n📁 Metrics saved as model_metrics.csv")


if __name__ == "__main__":
    main()
