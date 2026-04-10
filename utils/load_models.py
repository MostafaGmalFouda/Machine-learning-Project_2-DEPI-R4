import joblib

def load_all():
    models = {}
    metrics = {}

    for name in ["logistic", "random_forest", "gradient_boost", "xgboost"]:
        data = joblib.load(f"models/{name}.pkl")
        models[name] = data["model"]
        metrics[name] = data["metrics"]

    scaler = joblib.load("models/scaler.pkl")
    features = joblib.load("models/features.pkl")

    return models, scaler, features, metrics
