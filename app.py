import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="ML Assignment - Classification", layout="wide")

st.title("Machine Learning Classification App")
st.write("Upload test dataset and evaluate trained models.")


# ---------------------------
# Load Available Models
# ---------------------------
root = Path.cwd()
model_dir = root / "saved_models"

if not model_dir.exists():
    st.error("❌ saved_models folder not found.")
    st.stop()

model_files = list(model_dir.glob("*.pkl"))

if not model_files:
    st.error("❌ No trained models found.")
    st.stop()

model_names = [file.stem for file in model_files]

selected_model = st.selectbox("Select Model", model_names)


# ---------------------------
# Display Precomputed Metrics
# ---------------------------
metrics_path = root / "model_metrics.csv"

if metrics_path.exists():
    metrics_df = pd.read_csv(metrics_path)

    st.subheader("📊 Model Comparison Table")
    st.dataframe(metrics_df)

else:
    st.warning("⚠ model_metrics.csv not found. Run evaluate_models.py first.")


# ---------------------------
# Upload Test Dataset
# ---------------------------
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    if "diagnosis" not in df.columns:
        st.error("❌ Uploaded file must contain 'diagnosis' column.")
        st.stop()

    # Separate features and target
    X = df.drop(columns=["diagnosis"])
    y_true = df["diagnosis"]

    # Load selected model
    model_path = model_dir / f"{selected_model}.pkl"

    if not model_path.exists():
        st.error("❌ Selected model file not found.")
        st.stop()

    model = joblib.load(model_path)

    # Make predictions
    y_pred = model.predict(X)

    # ---------------------------
    # Compute Metrics
    # ---------------------------
    st.subheader(f"📈 Evaluation Metrics for {selected_model}")

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y_true, y_proba)
    else:
        auc = None

    metric_dict = {
        "Accuracy": acc,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    }

    metric_df = pd.DataFrame(metric_dict.items(), columns=["Metric", "Value"])
    st.dataframe(metric_df)


    # ---------------------------
    # Confusion Matrix
    # ---------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)


    # ---------------------------
    # Classification Report
    # ---------------------------
    st.subheader("Classification Report")

    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

else:
    st.info("Upload a CSV file to evaluate model performance.")
