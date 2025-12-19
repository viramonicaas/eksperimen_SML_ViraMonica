import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def main():
    # Load dataset
    train_df = pd.read_csv("dataset_preprocessing/diabetes_train_preprocessed.csv")
    test_df = pd.read_csv("dataset_preprocessing/diabetes_test_preprocessed.csv")

    X_train = train_df.drop(columns=["diabetes"])
    y_train = train_df["diabetes"]

    X_test = test_df.drop(columns=["diabetes"])
    y_test = test_df["diabetes"]

    # Hyperparameter space
    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "liblinear"]
    }

    # Manual MLflow logging
    mlflow.set_experiment("LogisticRegression_Tuning")

    for C in param_grid["C"]:
        for solver in param_grid["solver"]:
            with mlflow.start_run():
                # Log parameters
                mlflow.log_param("C", C)
                mlflow.log_param("solver", solver)
                mlflow.log_param("max_iter", 1000)

                # Train model
                model = LogisticRegression(
                    C=C,
                    solver=solver,
                    max_iter=1000
                )
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)

                # Metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                # Log metrics
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1_score", f1)

                # Log model
                mlflow.sklearn.log_model(model, "model")

                print(
                    f"C={C}, solver={solver} | "
                    f"Acc={acc:.4f}, F1={f1:.4f}"
                )


if __name__ == "__main__":
    main()