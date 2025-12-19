import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os


def preprocess_data(
    input_path: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Melakukan preprocessing dataset diabetes secara otomatis
    dan menyimpan hasilnya dalam bentuk CSV.
    """

    # Load dataset
    df = pd.read_csv(input_path)

    # Cleaning
    df = df.drop_duplicates()

    # Pisahkan fitur & target
    X = df.drop(columns=['diabetes', 'id'])
    y = df['diabetes']

    # Definisi fitur
    categorical_features = ['gender', 'physical_activity']

    numeric_features = [
        'age', 'bmi', 'blood_pressure', 'glucose',
        'cholesterol', 'heart_rate', 'sleep_hours',
        'smoking', 'alcohol_intake', 'family_history',
        'stress_level', 'diet_score', 'steps_per_day',
        'work_hours', 'water_intake_ltr', 'insulin'
    ]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # menerapkan preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    train_processed = X_train_df.copy()
    train_processed['diabetes'] = y_train.values

    test_processed = X_test_df.copy()
    test_processed['diabetes'] = y_test.values

    # Save output
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, 'diabetes_train_preprocessed.csv')
    test_path = os.path.join(output_dir, 'diabetes_test_preprocessed.csv')

    train_processed.to_csv(train_path, index=False)
    test_processed.to_csv(test_path, index=False)

    print("Preprocessing selesai.")
    print(f"Train saved to: {train_path}")
    print(f"Test saved to: {test_path}")


if __name__ == "__main__":
    preprocess_data(
        input_path="../dataset_raw/diabetes.csv",
        output_dir="./dataset_preprocessing"
    )