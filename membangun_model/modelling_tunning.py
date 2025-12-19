import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

mlflow.set_experiment("diabetes-advance")

df_train = pd.read_csv("dataset_preprocessing/diabetes_train_preprocessed.csv")
df_test = pd.read_csv("dataset_preprocessing/diabetes_test_preprocessed.csv")

X_train = df_train.drop(columns=["diabetes"])
y_train = df_train["diabetes"]
X_test = df_test.drop(columns=["diabetes"])
y_test = df_test["diabetes"]

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

    # Parameter
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 10)

    # Model
    mlflow.sklearn.log_model(model, "model")

    # Artifak
    joblib.dump(model, "rf_model.pkl")
    mlflow.log_artifact("rf_model.pkl")

    X_train.head().to_csv("sample_train.csv", index=False)
    mlflow.log_artifact("sample_train.csv")