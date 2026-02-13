# ML-Assignment - Classification Models with Streamlit

## Problem Statement
Build multiple classification models, evaluate them, and deploy an interactive Streamlit app.

The objective of this assignment is to implement multiple machine learning classification models on a real-world dataset, evaluate their performance using standard evaluation metrics, and deploy an interactive Streamlit web application for model testing and visualization.

The project demonstrates the complete machine learning workflow including data preprocessing, model training, performance evaluation, comparison, and deployment

## Dataset Description
This project uses the Breast Cancer Wisconsin (Diagnostic) Dataset

Source: UCI Machine Learning Repository

   >> Total Instances: 569
   >> Total Features: 30 numerical features
   >> Target Variable: Diagnosis
        M → Malignant (mapped to 1)
        B → Benign (mapped to 0)
   >> Problem Type: Binary Classification

The dataset contains computed features of digitized images of breast mass cell nuclei. These features describe characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

An 80-20 stratified train-test split was used to preserve class distribution.

## Models Used
1. Logistic Regression  
2. Decision Tree  
3. KNN  
4. Naive Bayes  
5. Random Forest  
6. XGBoost  

## Evaluation Metrics
Each model was evaluated using the following metrics:
    1. Accuracy
    2. AUC (Area Under ROC Curve)
    3. Precision
    4. Recall
    5. F1 Score
    6. Matthews Correlation Coefficient (MCC)

## Results Table

| ML Model            | Accuracy     | AUC          | Precision | Recall   | F1           | MCC          |
| ------------------- | ------------ | ------------ | --------- | -------- | ------------ | ------------ |
| Decision Tree       | 0.921053     | 0.944775     | 0.945946  | 0.833333 | 0.886076     | 0.829928     |
| KNN                 | 0.956140     | 0.983466     | 0.974359  | 0.904762 | 0.938272     | 0.905824     |
| Logistic Regression | **0.973684** | **0.996032** | 0.975610  | 0.952381 | **0.963855** | **0.943340** |
| Naive Bayes         | 0.938596     | 0.993386     | 1.000000  | 0.833333 | 0.909091     | 0.871489     |
| Random Forest       | 0.964912     | 0.994378     | 1.000000  | 0.904762 | 0.950000     | 0.925820     |
| XGBoost             | 0.964912     | 0.992063     | 1.000000  | 0.904762 | 0.950000     | 0.925820     |


## Observations

| Model | Key Strengths | Key Findings |
|-------|---------------|--------------|
| **Logistic Regression** | Highest Accuracy (97.37%), AUC (99.60%), F1 (96.39%), MCC (94.33%) | Best performer - dataset is linearly separable; simpler models work best |
| **Decision Tree** | Reasonable performance | Lower recall - some malignant cases missed; prone to overfitting |
| **KNN** | Strong performance (95.61% accuracy) | Works well with feature scaling; sensitive to distance metrics and k value |
| **Naive Bayes** | Perfect precision (100%) | More conservative predictions; lower recall affects overall performance |
| **Random Forest** | High precision (100%), improved stability | Strong but slightly below Logistic Regression; good feature importance insights |
| **XGBoost** | High precision (100%), strong AUC | Comparable to Random Forest; no significant improvement over simpler models |


## Streamlit deployed link
https://ml-assignment-2-ursiru6q2v9fmhmdhtldxj.streamlit.app/