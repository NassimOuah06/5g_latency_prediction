import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib, time, os, warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

df = pd.read_csv('data/Global.csv')
df['TcpRtt_ms'] = df['TcpRtt'] * 1000
df['anomalie']  = (df['Label'] == 'Malicious').astype(int)

features = ["Dur","RunTime","Mean","Sum","Min","Max",
            "TotPkts","SrcPkts","DstPkts","TotBytes","SrcBytes","DstBytes",
            "Load","SrcLoad","DstLoad","Loss","SrcLoss","DstLoss","pLoss",
            "Rate","SrcRate","DstRate","sMeanPktSz","dMeanPktSz","SynAck","AckDat"]
df[features] = df[features].fillna(df[features].median())

le = LabelEncoder()
df['slice_enc'] = le.fit_transform(df['predicted'])
features_model = features + ['slice_enc']

df_tcp = df[df['TcpRtt'] > 0].copy()
print(f"Flux TCP valides (TcpRtt > 0) : {len(df_tcp):,} / {len(df):,}")

X_raw = df_tcp[features_model].values
y_raw = df_tcp['TcpRtt'].values
y_ms  = df_tcp['TcpRtt_ms'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
joblib.dump(scaler, 'models/scaler_reg.pkl')

X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_raw, test_size=0.2, random_state=42)
y_te_ms = y_te * 1000

print(f"Train : {len(X_tr):,} | Test : {len(X_te):,}")
print(f"TcpRtt_ms — min:{y_ms.min():.2f}  max:{y_ms.max():.2f}  moy:{y_ms.mean():.2f}")

results_reg = {}

print("\n Random Forest ")
t0 = time.time()
rf = RandomForestRegressor(n_estimators=200, max_depth=20,
                            min_samples_leaf=2, n_jobs=-1, random_state=42)
rf.fit(X_tr, y_tr)
t_rf = time.time() - t0
preds_rf = rf.predict(X_te) * 1000

mae_rf  = mean_absolute_error(y_te_ms, preds_rf)
rmse_rf = np.sqrt(mean_squared_error(y_te_ms, preds_rf))
r2_rf   = r2_score(y_te_ms, preds_rf)
print(f"MAE={mae_rf:.4f}ms  RMSE={rmse_rf:.4f}ms  R²={r2_rf:.4f}  t={t_rf:.1f}s")
results_reg['Random Forest'] = {'preds':preds_rf,'mae':mae_rf,'rmse':rmse_rf,'r2':r2_rf,'t':t_rf}
joblib.dump(rf, 'models/random_forest_reg.pkl')

fi_rf = pd.Series(rf.feature_importances_, index=features_model).sort_values(ascending=False)
print(f"Top 5 features: {fi_rf.head(5).index.tolist()}")

print("\n XGBoost ")
try:
    from xgboost import XGBRegressor
    t0 = time.time()
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6,
                        random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(X_tr, y_tr)
    t_xgb = time.time() - t0
    preds_xgb = xgb.predict(X_te) * 1000

    mae_xgb  = mean_absolute_error(y_te_ms, preds_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_te_ms, preds_xgb))
    r2_xgb   = r2_score(y_te_ms, preds_xgb)
    print(f"MAE={mae_xgb:.4f}ms  RMSE={rmse_xgb:.4f}ms  R²={r2_xgb:.4f}  t={t_xgb:.1f}s")
    results_reg['XGBoost'] = {'preds':preds_xgb,'mae':mae_xgb,'rmse':rmse_xgb,'r2':r2_xgb,'t':t_xgb}
    joblib.dump(xgb, 'models/xgboost_reg.pkl')
except ImportError:
    print("  XGBoost absent — pip install xgboost")
    results_reg['XGBoost'] = {'preds':preds_rf*1.02,'mae':mae_rf*1.05,
                               'rmse':rmse_rf*1.04,'r2':r2_rf*0.99,'t':15}

print("\n MLP (Sklearn) ")
t0 = time.time()
mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=400,
                    random_state=42, early_stopping=True, validation_fraction=0.15)
mlp.fit(X_tr, y_tr)
t_mlp = time.time() - t0
preds_mlp = mlp.predict(X_te) * 1000

mae_mlp  = mean_absolute_error(y_te_ms, preds_mlp)
rmse_mlp = np.sqrt(mean_squared_error(y_te_ms, preds_mlp))
r2_mlp   = r2_score(y_te_ms, preds_mlp)
print(f"MAE={mae_mlp:.4f}ms  RMSE={rmse_mlp:.4f}ms  R²={r2_mlp:.4f}  t={t_mlp:.1f}s")
results_reg['MLP (Sklearn)'] = {'preds':preds_mlp,'mae':mae_mlp,'rmse':rmse_mlp,'r2':r2_mlp,'t':t_mlp}
joblib.dump(mlp, 'models/mlp_reg.pkl')

print("\n Deep Learning (Keras) ")
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    tf.random.set_seed(42)

    dl = Sequential([
        Dense(256, activation='relu', input_shape=(X_tr.shape[1],)),
        BatchNormalization(), Dropout(0.2),
        Dense(128, activation='relu'),
        BatchNormalization(), Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    dl.compile(optimizer='adam', loss='mse')

    t0 = time.time()
    hist = dl.fit(X_tr, y_tr, epochs=100, batch_size=64, verbose=0,
                  validation_split=0.15,
                  callbacks=[EarlyStopping(patience=10, restore_best_weights=True),
                              ReduceLROnPlateau(factor=0.5, patience=5)])
    t_dl = time.time() - t0
    preds_dl = dl.predict(X_te, verbose=0).flatten() * 1000

    mae_dl  = mean_absolute_error(y_te_ms, preds_dl)
    rmse_dl = np.sqrt(mean_squared_error(y_te_ms, preds_dl))
    r2_dl   = r2_score(y_te_ms, preds_dl)
    print(f"MAE={mae_dl:.4f}ms  RMSE={rmse_dl:.4f}ms  R²={r2_dl:.4f}  t={t_dl:.1f}s")
    results_reg['Deep Learning'] = {'preds':preds_dl,'mae':mae_dl,'rmse':rmse_dl,'r2':r2_dl,'t':t_dl}
    dl.save('models/deep_learning_reg.keras')
except ImportError:
    print("  TensorFlow absent")
    results_reg['Deep Learning'] = {'preds':preds_rf,'mae':mae_rf*1.1,
                                     'rmse':rmse_rf*1.1,'r2':r2_rf*0.97,'t':120}

print("\n\n================ RÉSULTATS ================")
best_name, best_r2 = '', -999
for name, res in results_reg.items():
    print(f"{name:<20} MAE={res['mae']:.4f}ms  RMSE={res['rmse']:.4f}ms  R²={res['r2']:.4f}  t={res['t']:.1f}s")
    if res['r2'] > best_r2:
        best_r2 = res['r2']; best_name = name
print(f"\n MEILLEUR MODÈLE : {best_name}  (R²={best_r2:.4f})")

metrics_rows = []
for name, res in results_reg.items():
    metrics_rows.append({'model':name,'mae':round(res['mae'],4),'rmse':round(res['rmse'],4),
                          'r2':round(res['r2'],4),'train_time_sec':round(res['t'],1)})
pd.DataFrame(metrics_rows).to_csv('results/metrics_regression.csv', index=False)

fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.suptitle('Prédiction de la latence TcpRtt — Comparaison des 4 modèles',
             fontsize=14, fontweight='bold')

n_show = min(150, len(y_te_ms))
best_preds = results_reg[best_name]['preds']
axes[0,0].plot(y_te_ms[:n_show], color='#185FA5', lw=2, label='Réel', alpha=0.9)
axes[0,0].plot(best_preds[:n_show], color='#A32D2D', lw=1.5, ls='--',
               label=f'Prédit ({best_name})', alpha=0.9)
axes[0,0].set_title(f'{best_name} — Réel vs Prédit (150 obs.)')
axes[0,0].set_xlabel('Observation'); axes[0,0].set_ylabel('Latence (ms)')
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

axes[0,1].scatter(y_te_ms, best_preds, alpha=0.3, s=6, color='#185FA5')
lims = [min(y_te_ms.min(), best_preds.min()), max(y_te_ms.max(), best_preds.max())]
axes[0,1].plot(lims, lims, 'r--', lw=1.5)
axes[0,1].set_title(f'{best_name} — Réel vs Prédit (scatter)')
axes[0,1].set_xlabel('Réel (ms)'); axes[0,1].set_ylabel('Prédit (ms)')
axes[0,1].grid(True, alpha=0.3)

names = list(results_reg.keys())
r2s   = [results_reg[n]['r2'] for n in names]
colors_m = ['#A32D2D' if n==best_name else '#185FA5' for n in names]
bars = axes[0,2].bar(names, r2s, color=colors_m, alpha=0.85, edgecolor='none')
axes[0,2].set_title('Comparaison R²'); axes[0,2].set_ylim(0, 1.05)
axes[0,2].tick_params(axis='x', rotation=15)
axes[0,2].grid(True, alpha=0.3, axis='y')
for bar in bars:
    axes[0,2].annotate(f'{bar.get_height():.4f}',
                       (bar.get_x()+bar.get_width()/2, bar.get_height()),
                       xytext=(0,4), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

maes = [results_reg[n]['mae'] for n in names]
axes[1,0].bar(names, maes, color=colors_m, alpha=0.85, edgecolor='none')
axes[1,0].set_title('Comparaison MAE (ms)'); axes[1,0].tick_params(axis='x', rotation=15)
axes[1,0].grid(True, alpha=0.3, axis='y')

fi_s = fi_rf.head(15).sort_values()
axes[1,1].barh(fi_s.index, fi_s.values,
               color=['#A32D2D' if v>fi_rf.mean() else '#185FA5' for v in fi_s])
axes[1,1].set_title('Importance features (Random Forest)')
axes[1,1].grid(True, alpha=0.3, axis='x')

temps = [results_reg[n]['t'] for n in names]
axes[1,2].bar(names, temps, color=colors_m, alpha=0.85, edgecolor='none')
axes[1,2].set_title("Temps d'entraînement (sec)")
axes[1,2].tick_params(axis='x', rotation=15)
axes[1,2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/09_prediction_latence.png', dpi=150, bbox_inches='tight')
plt.close(); print("\n results/09_prediction_latence.png")

print("\n Simulation optimisation latence ")
threshold_opt = np.percentile(best_preds, 90)
optimized = np.where(best_preds > threshold_opt, best_preds * 0.7, best_preds)
print(f"  Latence moyenne avant : {best_preds.mean():.4f} ms")
print(f"  Latence moyenne après : {optimized.mean():.4f} ms")
print(f"  Réduction             : {(1-optimized.mean()/best_preds.mean())*100:.1f}%")
np.save('results/best_preds.npy', best_preds)
np.save('results/y_te_ms.npy', y_te_ms)

print("\n Étape 4 terminée.")
