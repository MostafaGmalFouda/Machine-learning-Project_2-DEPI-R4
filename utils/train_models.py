import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    features_final, income, test_size=0.2, random_state=0
)

# ================= SCALER =================
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# IMPORTANT: keep dataframe format
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# ================= MODEL (IMPORTANT FIX) =================
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)  # مثال واضح
model.fit(X_train_scaled, y_train)

# ================= SAVE BUNDLE =================
bundle = {
    "model": model,
    "scaler": scaler,
    "features": list(X_train.columns)
}

joblib.dump(bundle, "models/model_bundle.pkl")