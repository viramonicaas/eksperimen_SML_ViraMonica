import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    # =====================
    # 1. Load dataset
    # =====================
    train_df = pd.read_csv("dataset_preprocessing/diabetes_train_preprocessed.csv")
    test_df = pd.read_csv("dataset_preprocessing/diabetes_test_preprocessed.csv")

    X_train = train_df.drop(columns=["diabetes"])
    y_train = train_df["diabetes"]

    X_test = test_df.drop(columns=["diabetes"])
    y_test = test_df["diabetes"]

    # =====================
    # 2. Enable MLflow autolog
    # =====================
    mlflow.sklearn.autolog()

    # =====================
    # 3. Train model
    # =====================
    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # =====================
        # 4. Evaluation
        # =====================
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Accuracy:", acc)
        print("Precision:", prec)
        print("Recall:", rec)
        print("F1-score:", f1)


if __name__ == "__main__":
    main()
