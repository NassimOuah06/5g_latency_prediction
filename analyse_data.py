import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# 🔥 Deep Learning (Keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("./dataset/Global.csv", na_values=["NA", "?", "NaN"])

# =========================
# 2. CLEANING
# =========================
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include=[object]).columns
df[cat_cols] = df[cat_cols].fillna("Unknown")

# =========================
# 3. FEATURES + TARGET
# =========================
features = [
    "Dur", "RunTime", "Mean", "Sum", "Min", "Max",
    "TotPkts", "SrcPkts", "DstPkts",
    "TotBytes", "SrcBytes", "DstBytes",
    "Load", "SrcLoad", "DstLoad",
    "Loss", "SrcLoss", "DstLoss", "pLoss",
    "Rate", "SrcRate", "DstRate",
    "sMeanPktSz", "dMeanPktSz"
]

features = [f for f in features if f in df.columns]

X = df[features]
y = df["TcpRtt"]  # 🎯 Latence 5G

# =========================
# 4. OUTLIERS (OPTIONAL)
# =========================
iso = IsolationForest(contamination="auto", random_state=42)
mask = iso.fit_predict(X)

df = df[mask == 1]

X = df[features]
y = df["TcpRtt"]

# =========================
# 5. SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# 6. SCALING (IMPORTANT DL)
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 7. ML MODELS
# =========================
models = {
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),

    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    ),

    "MLP (Sklearn)": MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=400,
        random_state=42
    )
}

results = {}

# =========================
# 8. TRAIN ML MODELS
# =========================
for name, model in models.items():
    print(f"\n[TRAIN] {name}")

    if name == "MLP (Sklearn)":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    results[name] = preds

# =========================
# 9. DEEP LEARNING MODEL (Keras)
# =========================
print("\n[TRAIN] Deep Learning (Keras)")

dl_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

dl_model.compile(optimizer='adam', loss='mse')

history = dl_model.fit(
    X_train_scaled,
    y_train,
    epochs=30,
    batch_size=32,
    verbose=0,
    validation_split=0.2
)

dl_preds = dl_model.predict(X_test_scaled).flatten()

results["Deep Learning (Keras)"] = dl_preds

# =========================
# 10. EVALUATION
# =========================
print("\n================ RESULTS ================\n")

scores = {}

for name, preds in results.items():
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    scores[name] = r2

    print(f"{name}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")
    print("-" * 40)

# =========================
# 11. BEST MODEL
# =========================
best_model = max(scores, key=scores.get)

print("\n🏆 BEST MODEL:", best_model)

# =========================
# 12. 5G OPTIMISATION SIMULATION
# =========================
best_preds = results[best_model]

threshold = np.percentile(best_preds, 90)

optimized = np.where(best_preds > threshold, best_preds * 0.7, best_preds)

print("\n========== OPTIMISATION ==========")
print("Avant :", np.mean(best_preds))
print("Après :", np.mean(optimized))

# =========================
# 13. VISUALISATION
# =========================
plt.figure(figsize=(10,6))

plt.plot(y_test.values[:100], label="Real", linewidth=2)

for name, preds in results.items():
    plt.plot(preds[:100], label=name)

plt.legend()
plt.title("5G Latency Prediction - ML vs Deep Learning")
plt.savefig('5g_latency_prediction.png')
plt.show()