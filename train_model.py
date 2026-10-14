"""Train and evaluate a regression model on reproducible synthetic data."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 42

def create_dataset(n_samples=500):
    """Create synthetic student study data for a learning demo."""
    rng = np.random.default_rng(RANDOM_SEED)
    study_hours = rng.uniform(0, 10, n_samples)
    attendance = rng.uniform(40, 100, n_samples)
    previous_score = rng.uniform(25, 95, n_samples)
    noise = rng.normal(0, 5, n_samples)
    final_score = (
        4.2 * study_hours
        + 0.22 * attendance
        + 0.42 * previous_score
        + noise
        + 5
    )
    final_score = np.clip(final_score, 0, 100)
    return pd.DataFrame({
        "study_hours": study_hours.round(2),
        "attendance": attendance.round(2),
        "previous_score": previous_score.round(2),
        "final_score": final_score.round(2),
    })

def main():
    data = create_dataset()
    data_path = BASE_DIR / "student_scores_sample.csv"
    data.to_csv(data_path, index=False)

    features = ["study_hours", "attendance", "previous_score"]
    X = data[features]
    y = data["final_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": r2_score(y_test, predictions),
    }
    joblib.dump(model, BASE_DIR / "model.joblib")
    pd.DataFrame([metrics]).to_csv(BASE_DIR / "evaluation_metrics.csv", index=False)

    print("Training complete.")
    print(f"Dataset: {data_path.name} ({len(data)} rows)")
    print(f"Saved model: {(BASE_DIR / 'model.joblib').name}")
    print("Evaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.3f}")
    print("Note: Results are for synthetic educational data only.")

if __name__ == "__main__":
    main()
