import os
import joblib

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return metrics



def save_model_bundle(model, scaler, X_test_scaled, X_test_original, y_test, name):

    metrics = evaluate_model(
        model,
        X_test_scaled,
        y_test
    )

    os.makedirs("models", exist_ok=True)

    bundle = {
        "model": model,
        "scaler": scaler,
        "features": list(X_test_original.columns),
        "metrics": metrics
    }

    joblib.dump(
        bundle,
        f"models/{name}.pkl"
    )

    print(f"{name}.pkl saved")

    return metrics