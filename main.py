import subprocess, sys

def run(script, title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")
    r = subprocess.run([sys.executable, script])
    if r.returncode != 0: print(f"  Erreur dans {script}")
    else: print(" Terminé.")

if __name__ == '__main__':
    print("""

  PROJET TER M1 RSA — Optimisation 5G par IA      
  Dataset : Global.csv — 14 456 flux réseau — 52 colonnes    

    """)
    run('etape1_dataset.py',            'ÉTAPE 1 — Dataset')
    run('etape2_eda.py',                'ÉTAPE 2 — EDA')
    run('etape3_anomalies.py',          'ÉTAPE 3 — Détection anomalies')
    run('etape4_prediction_latence.py', 'ÉTAPE 4 — Prédiction latence')
    run('etape5_optimisation.py',       'ÉTAPE 5 — Optimisation ressources')

    print("""
 PROJET COMPLET TERMINÉ

Figures dans results/ :
  01_latence_distribution.png   Distributions latence
  02_features_distribution.png  Features Benign vs Malicious
  03_correlation.png            Heatmap corrélations
  04_correlation_target.png     Corrélation → TcpRtt
  05_stats_slice.png            Stats par slice
  06_pca.png                    PCA 2D
  07_detection_anomalies.png    Détection anomalies (IF / RF / DL)
  09_prediction_latence.png     Prédiction — 4 modèles
  10_optimisation.png           Optimisation QoS

    """)
