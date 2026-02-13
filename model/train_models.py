import os
import pandas as pd
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def load_wdbc_data():
    if os.path.exists("data/wdbc.data"):
        file_path = "data/wdbc.data"
    else:
        file_path = os.path.join("..", "data", "wdbc.data")
    
    file_path = os.path.abspath(file_path)
    print(os.getcwd())
    print("Loading dataset from:", file_path)

    df = pd.read_csv(file_path, header=None)
    print("✅ Dataset loaded successfully. Shape:", df.shape)

    feature_names = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]

    df.columns = ["id", "diagnosis"] + feature_names
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0}) # M=1, B=0  

    return df


def main():
    df = load_wdbc_data()
    X = df.drop(columns=["id", "diagnosis"])
    y = df["diagnosis"]

    # ✅ Train-test split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

    # ✅ Create folders
    root = Path.cwd()
    save_dir = root / "saved_models"
    save_dir.mkdir(parents=True, exist_ok=True)

    data_dir = ".."/ root / "data"
    print("Data directory:", data_dir)
    data_dir.mkdir(exist_ok=True)

    # ✅ Save datasets
    train_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    test_data = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)
    train_data.to_csv(data_dir / "train_data.csv", index=False)
    test_data.to_csv(data_dir / "test_data.csv", index=False)
    print("✅ Datasets saved successfully.")

    # ✅ Define 6 Models 
    models = {
        "logistic_regression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, C=1.2, solver="liblinear"))]),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "knn": Pipeline([("scaler", StandardScaler()),("clf", KNeighborsClassifier(n_neighbors=7, weights="distance"))]),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(n_estimators=250, max_depth=8, random_state=42),
        "xg_boost": XGBClassifier(n_estimators=250, learning_rate=0.07, max_depth=4, subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric="logloss")
    }

    # ✅ Train + Save models
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, save_dir / f"{model_name}.pkl")
        print(f"✅ Saved model: {model_name}.pkl")


if __name__ == "__main__":
    main()

