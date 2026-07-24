import os
import joblib


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_all():

    model_names = [
        "logistic",
        "random_forest",
        "gradient_boost",
        "xgboost"
    ]

    models = {}
    metrics = {}

    scaler = None
    features = None


    for name in model_names:

        path = os.path.join(
            MODEL_DIR,
            f"{name}.pkl"
        )

        bundle = joblib.load(path)

        models[name] = bundle["model"]

        scaler = bundle["scaler"]
        features = bundle["features"]

        metrics[name] = bundle.get(
            "metrics",
            {}
        )


    return models, scaler, features, metrics