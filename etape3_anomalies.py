import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_curve)
import joblib, time, os, warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

df = pd.read_csv('data/Global.csv')
df['TcpRtt_ms'] = df['TcpRtt'] * 1000
df['anomalie']  = (df['Label'] == 'Malicious').astype(int)
df['slice']     = df['predicted'].map({'1:eMBB':'eMBB','2:mMTC':'mMTC','3:URLLC':'URLLC'})

features = ["Dur","RunTime","Mean","Sum","Min","Max",
            "TotPkts","SrcPkts","DstPkts","TotBytes","SrcBytes","DstBytes",
            "Load","SrcLoad","DstLoad","Loss","SrcLoss","DstLoss","pLoss",
            "Rate","SrcRate","DstRate","sMeanPktSz","dMeanPktSz","SynAck","AckDat"]
df[features] = df[features].fillna(df[features].median())

le = LabelEncoder()
df['slice_enc'] = le.fit_transform(df['predicted'])
feat_model = features + ['slice_enc']

y  = df['anomalie'].values
sc = StandardScaler()
X  = sc.fit_transform(df[feat_model])
joblib.dump(sc, 'models/scaler_anomaly.pkl')

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print(f"Dataset  : {X.shape[0]:,} obs × {X.shape[1]} features")
print(f"Benign   : {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
print(f"Malicious: {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")
print(f"Train : {len(X_tr):,}  |  Test : {len(X_te):,}")

results_clf = {}

print("\n" + "="*55)
print("MODÈLE A — ISOLATION FOREST (non supervisé)")
print("="*55)

t0 = time.time()
iso = IsolationForest(n_estimators=300, contamination=0.49,
                      max_features=1.0, random_state=42, n_jobs=-1)
iso.fit(X_tr)
t_if = time.time() - t0
y_pred_if = (iso.predict(X_te) == -1).astype(int)
scores_if  = -iso.decision_function(X_te)

f1_if  = f1_score(y_te, y_pred_if)
pr_if  = precision_score(y_te, y_pred_if)
re_if  = recall_score(y_te, y_pred_if)
auc_if = roc_auc_score(y_te, scores_if)
print(classification_report(y_te, y_pred_if, target_names=['Benign','Malicious']))
print(f"AUC-ROC : {auc_if:.4f}  |  Temps : {t_if:.2f}s")
results_clf['Isolation Forest'] = {'y_pred':y_pred_if,'scores':scores_if,
    'f1':f1_if,'pr':pr_if,'re':re_if,'auc':auc_if,'t':t_if}
joblib.dump(iso, 'models/isolation_forest.pkl')

print("\n" + "="*55)
print("MODÈLE B — RANDOM FOREST CLASSIFIER (supervisé)")
print("="*55)

t0 = time.time()
rfc = RandomForestClassifier(n_estimators=200, max_depth=20,
                              n_jobs=-1, random_state=42)
rfc.fit(X_tr, y_tr)
t_rfc = time.time() - t0
y_pred_rfc  = rfc.predict(X_te)
scores_rfc  = rfc.predict_proba(X_te)[:,1]

f1_rfc  = f1_score(y_te, y_pred_rfc)
pr_rfc  = precision_score(y_te, y_pred_rfc)
re_rfc  = recall_score(y_te, y_pred_rfc)
auc_rfc = roc_auc_score(y_te, scores_rfc)
print(classification_report(y_te, y_pred_rfc, target_names=['Benign','Malicious']))
print(f"AUC-ROC : {auc_rfc:.4f}  |  Temps : {t_rfc:.2f}s")
results_clf['Random Forest'] = {'y_pred':y_pred_rfc,'scores':scores_rfc,
    'f1':f1_rfc,'pr':pr_rfc,'re':re_rfc,'auc':auc_rfc,'t':t_rfc}
joblib.dump(rfc, 'models/rf_classifier.pkl')

fi = pd.Series(rfc.feature_importances_, index=feat_model).sort_values(ascending=False)
print(f"Top 5 features : {fi.head(5).index.tolist()}")
np.save('results/y_true.npy', y_te)
np.save('results/scores_rf.npy', scores_rfc)

print("\n" + "="*55)
print("MODÈLE C — DEEP LEARNING (Keras)")
print("="*55)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    tf.random.set_seed(42)

    dl = Sequential([
        Dense(256, activation='relu', input_shape=(X_tr.shape[1],)),
        BatchNormalization(), Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(), Dropout(0.2),
        Dense(64,  activation='relu'),
        Dense(32,  activation='relu'),
        Dense(1,   activation='sigmoid')
    ])
    dl.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    t0 = time.time()
    hist = dl.fit(X_tr, y_tr, epochs=50, batch_size=64, verbose=0,
                  validation_split=0.15,
                  callbacks=[EarlyStopping(patience=8, restore_best_weights=True),
                              ReduceLROnPlateau(factor=0.5, patience=4)])
    t_dl = time.time() - t0
    scores_dl  = dl.predict(X_te, verbose=0).flatten()
    y_pred_dl  = (scores_dl > 0.5).astype(int)

    f1_dl  = f1_score(y_te, y_pred_dl)
    pr_dl  = precision_score(y_te, y_pred_dl)
    re_dl  = recall_score(y_te, y_pred_dl)
    auc_dl = roc_auc_score(y_te, scores_dl)
    print(classification_report(y_te, y_pred_dl, target_names=['Benign','Malicious']))
    print(f"AUC-ROC : {auc_dl:.4f}  |  Temps : {t_dl:.1f}s")
    results_clf['Deep Learning'] = {'y_pred':y_pred_dl,'scores':scores_dl,
        'f1':f1_dl,'pr':pr_dl,'re':re_dl,'auc':auc_dl,'t':t_dl,
        'history':hist.history}
    dl.save('models/dl_classifier.keras')
    np.save('results/scores_dl.npy', scores_dl)

except ImportError:
    print("  TensorFlow absent — pip install tensorflow")
    print("   Utilisation MLP Sklearn à la place...")
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(256,128,64,32), max_iter=200,
                        random_state=42, early_stopping=True, validation_fraction=0.15)
    mlp.fit(X_tr, y_tr)
    t_dl = time.time() - t0
    scores_dl  = mlp.predict_proba(X_te)[:,1]
    y_pred_dl  = mlp.predict(X_te)
    f1_dl  = f1_score(y_te, y_pred_dl)
    pr_dl  = precision_score(y_te, y_pred_dl)
    re_dl  = recall_score(y_te, y_pred_dl)
    auc_dl = roc_auc_score(y_te, scores_dl)
    print(classification_report(y_te, y_pred_dl, target_names=['Benign','Malicious']))
    print(f"AUC-ROC : {auc_dl:.4f}  |  Temps : {t_dl:.1f}s")
    results_clf['MLP Classifier'] = {'y_pred':y_pred_dl,'scores':scores_dl,
        'f1':f1_dl,'pr':pr_dl,'re':re_dl,'auc':auc_dl,'t':t_dl}
    np.save('results/scores_dl.npy', scores_dl)

print("\n\n================ RÉSULTATS DÉTECTION ================")
best_name, best_f1 = '', 0
for name, res in results_clf.items():
    print(f"{name:<20} F1={res['f1']:.4f}  Prec={res['pr']:.4f}  "
          f"Recall={res['re']:.4f}  AUC={res['auc']:.4f}  t={res['t']:.1f}s")
    if res['f1'] > best_f1:
        best_f1 = res['f1']; best_name = name
print(f"\n MEILLEUR : {best_name}  (F1={best_f1:.4f})")

pd.DataFrame([{'model':n,'f1':round(v['f1'],4),'precision':round(v['pr'],4),
               'recall':round(v['re'],4),'auc_roc':round(v['auc'],4),
               'train_time_sec':round(v['t'],1)} for n,v in results_clf.items()]
             ).to_csv('results/metrics_anomaly.csv', index=False)

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle('Détection d\'anomalies réseau 5G — Comparaison des modèles',
             fontsize=14, fontweight='bold')

colors_m = ['#BA7517','#3B6D11','#185FA5','#A32D2D']
for i, (name, res) in enumerate(results_clf.items()):
    fpr, tpr, _ = roc_curve(y_te, res['scores'])
    axes[0,0].plot(fpr, tpr, color=colors_m[i], lw=2,
                   label=f"{name} (AUC={res['auc']:.3f})")
axes[0,0].plot([0,1],[0,1],'k--',lw=1)
axes[0,0].set_title('Courbes ROC superposées')
axes[0,0].set_xlabel('Faux Positifs'); axes[0,0].set_ylabel('Vrais Positifs')
axes[0,0].legend(fontsize=9); axes[0,0].grid(True, alpha=0.3)

names = list(results_clf.keys())
f1s   = [results_clf[n]['f1']  for n in names]
aucs  = [results_clf[n]['auc'] for n in names]
x = np.arange(len(names)); w = 0.35
axes[0,1].bar(x-w/2, f1s,  w, label='F1-Score',  color='#185FA5', alpha=0.85, edgecolor='none')
axes[0,1].bar(x+w/2, aucs, w, label='AUC-ROC',   color='#3B6D11', alpha=0.85, edgecolor='none')
axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(names, rotation=10, fontsize=9)
axes[0,1].set_ylim(0, 1.1); axes[0,1].set_title('F1-Score et AUC-ROC')
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3, axis='y')
for bars in [axes[0,1].patches[:len(names)], axes[0,1].patches[len(names):]]:
    for bar in bars:
        if bar.get_height() > 0.01:
            axes[0,1].annotate(f'{bar.get_height():.3f}',
                               (bar.get_x()+bar.get_width()/2, bar.get_height()),
                               xytext=(0,3), textcoords='offset points',
                               ha='center', fontsize=8)

cm = confusion_matrix(y_te, results_clf[best_name]['y_pred'])
axes[0,2].imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        axes[0,2].text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                       fontsize=14, fontweight='bold',
                       color='white' if cm[i,j] > cm.max()/2 else 'black')
axes[0,2].set_xticks([0,1]); axes[0,2].set_yticks([0,1])
axes[0,2].set_xticklabels(['Prédit Benign','Prédit Malicious'])
axes[0,2].set_yticklabels(['Réel Benign','Réel Malicious'])
axes[0,2].set_title(f'Confusion — {best_name}')

fi_s = fi.head(15).sort_values()
axes[1,0].barh(fi_s.index, fi_s.values,
               color=['#A32D2D' if v > fi.mean() else '#185FA5' for v in fi_s])
axes[1,0].set_title('Importance features (Random Forest)')
axes[1,0].grid(True, alpha=0.3, axis='x')

axes[1,1].hist(results_clf[best_name]['scores'][y_te==0], bins=60,
               alpha=0.65, label='Benign', color='#3B6D11', edgecolor='none')
axes[1,1].hist(results_clf[best_name]['scores'][y_te==1], bins=60,
               alpha=0.65, label='Malicious', color='#A32D2D', edgecolor='none')
axes[1,1].set_title(f'Distribution scores — {best_name}')
axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

temps = [results_clf[n]['t'] for n in names]
axes[1,2].bar(names, temps,
              color=['#BA7517','#3B6D11','#185FA5','#A32D2D'][:len(names)],
              alpha=0.85, edgecolor='none')
axes[1,2].set_title("Temps d'entraînement (sec)")
axes[1,2].tick_params(axis='x', rotation=10)
axes[1,2].grid(True, alpha=0.3, axis='y')
for i, (bar, t) in enumerate(zip(axes[1,2].patches, temps)):
    axes[1,2].annotate(f'{t:.1f}s',
                       (bar.get_x()+bar.get_width()/2, bar.get_height()),
                       xytext=(0,3), textcoords='offset points', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('results/07_detection_anomalies.png', dpi=150, bbox_inches='tight')
plt.close()
print(" results/07_detection_anomalies.png")

df_te = df.iloc[len(df) - len(y_te):].copy()
df_te['pred'] = results_clf[best_name]['y_pred']
print(f"\n Résultats {best_name} par slice ")
for sl in ['eMBB','mMTC','URLLC']:
    s = df_te[df_te.slice==sl]
    if len(s) == 0: continue
    tp = ((s.pred==1)&(s.anomalie==1)).sum()
    fn = ((s.pred==0)&(s.anomalie==1)).sum()
    fp = ((s.pred==1)&(s.anomalie==0)).sum()
    f1s = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0
    print(f"  {sl:<6}: TP={tp:>4} FN={fn:>4} FP={fp:>4} | F1={f1s:.3f}")

print("\n Étape 3 terminée.")
