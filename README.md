# Optimisation 5G par IA

## Source du dataset
**5G-SliciNdd** — Figshare  
DOI : [10.6084/m9.figshare.24446515](https://doi.org/10.6084/m9.figshare.24446515)  


## Dataset utilisé : Global.csv

| Colonne   | Description                           |
|-----------|---------------------------------------|
| TcpRtt    | Latence TCP RTT (s)                   |
| SynAck    | Délai SYN→ACK (s)                     |
| AckDat    | Délai ACK→Data (s)                    |
| Dur       | Durée du flux (s)                     |
| TotBytes  | Volume total (octets)                 |
| Load      | Charge réseau (bps)                   |
| pLoss     | Taux perte paquets (%)                |
| predicted | Slice : 1=eMBB / 2=mMTC / 3=URLLC     |
| Label     | Benign / Malicious ← CIBLE détection  |

- **14 456 flux** | **52 colonnes** | **51.2 % Malicious**
- Slices : eMBB (5 808) · mMTC (4 615) · URLLC (4 033)

## Structure

```
projet_final/
├── main.py                        ← Lance tout (étapes 1 à 5)
├── etape1_dataset.py              ← Description et statistiques du dataset
├── etape2_eda.py                  ← EDA (6 graphiques → results/)
├── etape3_anomalies.py            ← IF + RF + Deep Learning 
├── etape4_prediction_latence.py   ← RF + XGBoost + MLP + Deep Learning
├── etape5_optimisation.py         ← Optimisation QoS        
├── requirements.txt
└── data/
    ├── Global.csv   ← dataset principal (14 456 flux)
    ├── URLLC.csv
    ├── eMBB.csv
    └── mMTC.csv
```

## Installation & lancement

```bash
# Python 3.10 à 3.12 recommandé (compatibilité TensorFlow)
pip install -r requirements.txt
python main.py
```

## Fichiers générés après exécution

```
results/
├── 01_latence_distribution.png
├── 02_features_distribution.png
├── 03_correlation.png
├── 04_correlation_target.png
├── 05_stats_slice.png
├── 06_pca.png
├── 07_detection_anomalies.png
├── 09_prediction_latence.png
├── 10_optimisation.png
├── metrics_anomaly.csv
├── metrics_regression.csv
└── recommandations.csv
models/
├── rf_classifier.pkl
├── random_forest_reg.pkl
├── xgboost_reg.pkl
├── mlp_reg.pkl
├── isolation_forest.pkl
├── dl_classifier.keras
├── deep_learning_reg.keras
├── scaler_anomaly.pkl
└── scaler_reg.pkl
```
