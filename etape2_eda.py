import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

os.makedirs('results', exist_ok=True)

df = pd.read_csv('data/Global.csv')
df['TcpRtt_ms'] = df['TcpRtt'] * 1000
df['anomalie']  = (df['Label'] == 'Malicious').astype(int)
df['slice']     = df['predicted'].map({'1:eMBB':'eMBB','2:mMTC':'mMTC','3:URLLC':'URLLC'})

features = ["Dur","RunTime","Mean","Sum","Min","Max",
            "TotPkts","SrcPkts","DstPkts","TotBytes","SrcBytes","DstBytes",
            "Load","SrcLoad","DstLoad","Loss","SrcLoss","DstLoss","pLoss",
            "Rate","SrcRate","DstRate","sMeanPktSz","dMeanPktSz","SynAck","AckDat"]
df[features] = df[features].fillna(df[features].median())

print("ÉTAPE 2 — ANALYSE EXPLORATOIRE")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Distribution de la latence TcpRtt (ms) par Label et par Slice',
             fontsize=13, fontweight='bold')

axes[0].hist(df[df.anomalie==0]['TcpRtt_ms'].clip(0,200), bins=60,
             alpha=0.65, label='Benign', color='#185FA5', edgecolor='none')
axes[0].hist(df[df.anomalie==1]['TcpRtt_ms'].clip(0,200), bins=60,
             alpha=0.65, label='Malicious', color='#A32D2D', edgecolor='none')
axes[0].set_title('Global (tronqué à 200ms)'); axes[0].set_xlabel('Latence (ms)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

slice_colors = {'eMBB':'#185FA5','mMTC':'#BA7517','URLLC':'#3B6D11'}
for sl, c in slice_colors.items():
    axes[1].hist(df[df.slice==sl]['TcpRtt_ms'].clip(0,200), bins=50,
                 alpha=0.6, label=sl, color=c, edgecolor='none')
axes[1].set_title('Par type de slice'); axes[1].set_xlabel('Latence (ms)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

data_box = [df[(df.slice==s)&(df.anomalie==l)]['TcpRtt_ms'].clip(0,300).values
            for s in ['eMBB','mMTC','URLLC'] for l in [0,1]]
bp = axes[2].boxplot(data_box, patch_artist=True,
                     medianprops=dict(color='white', linewidth=2))
colors_bp = ['#185FA5','#A32D2D','#BA7517','#E87A40','#3B6D11','#6BAF3A']
for patch, c in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(c); patch.set_alpha(0.75)
axes[2].set_xticklabels(['eMBB\nBenign','eMBB\nMalic.','mMTC\nBenign',
                          'mMTC\nMalic.','URLLC\nBenign','URLLC\nMalic.'], fontsize=8)
axes[2].set_title('Boxplots slice × label'); axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/01_latence_distribution.png', dpi=150, bbox_inches='tight')
plt.close(); print(" results/01_latence_distribution.png")

feat_show = ['SynAck','AckDat','Dur','TotBytes','Load','pLoss','Loss','Rate']
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle('Distribution des features : Benign vs Malicious', fontsize=13, fontweight='bold')
for i, f in enumerate(feat_show):
    ax = axes[i//4][i%4]
    vals_b = df[df.anomalie==0][f].clip(df[f].quantile(0.01), df[f].quantile(0.99))
    vals_m = df[df.anomalie==1][f].clip(df[f].quantile(0.01), df[f].quantile(0.99))
    ax.hist(vals_b, bins=50, alpha=0.65, label='Benign', color='#185FA5', edgecolor='none')
    ax.hist(vals_m, bins=50, alpha=0.65, label='Malicious', color='#A32D2D', edgecolor='none')
    ax.set_title(f, fontweight='bold', fontsize=10)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/02_features_distribution.png', dpi=150, bbox_inches='tight')
plt.close(); print(" results/02_features_distribution.png")

top_feat = features[:16] + ['TcpRtt_ms','anomalie']
plt.figure(figsize=(14, 10))
corr = df[top_feat].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            mask=mask, linewidths=0.4, annot_kws={'size':8}, vmin=-1, vmax=1)
plt.title('Matrice de corrélation — Features réseau 5G', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/03_correlation.png', dpi=150, bbox_inches='tight')
plt.close(); print(" results/03_correlation.png")

corr_target = df[features + ['TcpRtt_ms']].corr()['TcpRtt_ms'].drop('TcpRtt_ms')
corr_target = corr_target.sort_values(key=abs, ascending=True)
plt.figure(figsize=(10, 8))
colors_bar = ['#A32D2D' if v > 0 else '#185FA5' for v in corr_target]
plt.barh(corr_target.index, corr_target.values, color=colors_bar, alpha=0.85)
plt.axvline(0, color='black', lw=0.8)
plt.title('Corrélation des features avec TcpRtt_ms (latence)', fontsize=12, fontweight='bold')
plt.xlabel('Coefficient de corrélation de Pearson')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('results/04_correlation_target.png', dpi=150, bbox_inches='tight')
plt.close(); print(" results/04_correlation_target.png")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Analyse par type de slice 5G (eMBB / mMTC / URLLC)', fontsize=13, fontweight='bold')
for i, feat in enumerate(['TcpRtt_ms','SynAck','Load','TotBytes','pLoss','Rate']):
    ax = axes[i//3][i%3]
    data_s = [df[df.slice==s][feat].clip(df[feat].quantile(0.01),
               df[feat].quantile(0.99)).values for s in ['eMBB','mMTC','URLLC']]
    bp2 = ax.boxplot(data_s, patch_artist=True, medianprops=dict(color='white',linewidth=2))
    for patch, c in zip(bp2['boxes'], ['#185FA5','#BA7517','#3B6D11']):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.set_xticklabels(['eMBB','mMTC','URLLC'])
    ax.set_title(feat, fontweight='bold'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/05_stats_slice.png', dpi=150, bbox_inches='tight')
plt.close(); print(" results/05_stats_slice.png")

Xs = StandardScaler().fit_transform(df[features])
pca = PCA(n_components=2, random_state=42)
Xp = pca.fit_transform(Xs)
plt.figure(figsize=(10, 7))
mn = df['anomalie'].values == 0; ma = ~mn
plt.scatter(Xp[mn,0], Xp[mn,1], c='#185FA5', alpha=0.2, s=6, label=f'Benign ({mn.sum()})')
plt.scatter(Xp[ma,0], Xp[ma,1], c='#A32D2D', alpha=0.5, s=10,
            label=f'Malicious ({ma.sum()})', marker='x')
plt.title(f'PCA 2D — Variance expliquée : {pca.explained_variance_ratio_.sum()*100:.1f}%',
          fontsize=13, fontweight='bold')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.legend(fontsize=11); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/06_pca.png', dpi=150, bbox_inches='tight')
plt.close(); print(" results/06_pca.png")

print("\n Top corrélations avec TcpRtt_ms ")
print(corr_target.sort_values(key=abs, ascending=False).head(10).round(4).to_string())
print("\n Outliers IQR ")
for f in ['TcpRtt_ms','SynAck','AckDat','Load','pLoss']:
    Q1,Q3 = df[f].quantile([.25,.75])
    n = ((df[f]<Q1-1.5*(Q3-Q1))|(df[f]>Q3+1.5*(Q3-Q1))).sum()
    print(f"  {f:<15}: {n:>5} ({n/len(df)*100:.2f}%)")
print("\n Étape 2 terminée.")
