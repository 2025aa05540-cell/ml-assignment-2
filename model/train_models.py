from sklearn.datasets import load_breast_cancer


def main():
    print("Load dataset from kaggle and train classification models here.")
    data = load_breast_cancer()
    X, y = data.data, data.target
    print(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")
    
