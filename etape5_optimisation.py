import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

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

print("=" * 60)
print("ÉTAPE 5 — OPTIMISATION & RECOMMANDATIONS QoS")
print("=" * 60)

sla_thresholds = {
    'URLLC': {'TcpRtt_ms': 1.0,   'pLoss': 0.001, 'label': 'Ultra faible latence'},
    'eMBB':  {'TcpRtt_ms': 30.0,  'pLoss': 1.0,   'label': 'Haut débit'},
    'mMTC':  {'TcpRtt_ms': 100.0, 'pLoss': 5.0,   'label': 'IoT massif'},
}

print("\n Violations SLA par slice (TcpRtt) ")
for sl, thresh in sla_thresholds.items():
    sub = df[df.slice == sl]
    n_viol_lat = (sub['TcpRtt_ms'] > thresh['TcpRtt_ms']).sum()
    n_viol_pkt = (sub['pLoss'] > thresh['pLoss']).sum()
    n_total     = len(sub)
    print(f"  {sl:<6} ({thresh['label']}) :")
    print(f"    Latence > {thresh['TcpRtt_ms']}ms : {n_viol_lat:>4} / {n_total} ({n_viol_lat/n_total*100:.1f}%)")
    print(f"    pLoss > {thresh['pLoss']}%        : {n_viol_pkt:>4} / {n_total} ({n_viol_pkt/n_total*100:.1f}%)")

def get_recommendations(row):
    sl = row.get('slice', 'Unknown')
    thresh = sla_thresholds.get(sl, {})
    recs = []

    if thresh:
        if row['TcpRtt_ms'] > thresh['TcpRtt_ms']:
            recs.append(f" Latence {row['TcpRtt_ms']:.2f}ms > {thresh['TcpRtt_ms']}ms : "
                         "allouer ressources PRB prioritaires")
        if row['pLoss'] > thresh['pLoss']:
            recs.append(f" Pertes {row['pLoss']:.3f}% > {thresh['pLoss']}% : "
                         "activer HARQ / retransmission adaptative")

    if row['Load'] > df['Load'].quantile(0.9):
        recs.append(" Charge réseau élevée : load balancing intercellulaire")
    if row['anomalie'] == 1:
        recs.append(" Flux malveillant détecté : isoler et bloquer le flux")
    if row['Loss'] > df['Loss'].quantile(0.95):
        recs.append(" Pertes élevées : vérifier qualité de lien radio")

    return recs if recs else [" Flux conforme — aucune action requise"]

anomalies_sample = df[df.anomalie == 1].head(20)
print("\n Recommandations sur 20 flux malveillants ")
rec_rows = []
for _, row in anomalies_sample.iterrows():
    recs = get_recommendations(row)
    for r in recs:
        rec_rows.append({'slice':row.get('slice','?'),'TcpRtt_ms':round(row['TcpRtt_ms'],3),
                          'pLoss':round(row['pLoss'],3),'label':row['Label'],'recommandation':r})

rec_df = pd.DataFrame(rec_rows)
rec_df.to_csv('results/recommandations.csv', index=False)
print(rec_df[['slice','TcpRtt_ms','recommandation']].head(10).to_string(index=False))

print("\n Simulation : impact des optimisations ")
df_opt = df.copy()

mask_urllc_viol = (df_opt.slice=='URLLC') & (df_opt['TcpRtt_ms'] > 1.0)
df_opt.loc[mask_urllc_viol, 'TcpRtt_ms'] *= 0.6

mask_loss = df_opt['pLoss'] > 1.0
df_opt.loc[mask_loss, 'pLoss'] *= 0.4

mask_mal = df_opt['anomalie'] == 1
df_opt.loc[mask_mal, 'TcpRtt_ms'] *= 0.8

for sl in ['eMBB','mMTC','URLLC']:
    av = df[df.slice==sl]['TcpRtt_ms'].mean()
    ap = df_opt[df_opt.slice==sl]['TcpRtt_ms'].mean()
    print(f"  {sl:<6}: {av:.2f}ms → {ap:.2f}ms  (−{(av-ap)/av*100:.1f}%)")

av_global = df['TcpRtt_ms'].mean()
ap_global = df_opt['TcpRtt_ms'].mean()
print(f"  Global: {av_global:.2f}ms → {ap_global:.2f}ms  (−{(av_global-ap_global)/av_global*100:.1f}%)")

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle('Optimisation des ressources 5G — Analyse et recommandations',
             fontsize=14, fontweight='bold')

slices_list = ['eMBB','mMTC','URLLC']
lat_av = [df[df.slice==s]['TcpRtt_ms'].mean() for s in slices_list]
lat_ap = [df_opt[df_opt.slice==s]['TcpRtt_ms'].mean() for s in slices_list]
x = np.arange(3); w = 0.35
b1 = axes[0,0].bar(x-w/2, lat_av, w, label='Avant', color='#A32D2D', alpha=0.85, edgecolor='none')
b2 = axes[0,0].bar(x+w/2, lat_ap, w, label='Après', color='#3B6D11', alpha=0.85, edgecolor='none')
axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(slices_list)
axes[0,0].set_title('Latence moy. avant/après (ms)')
axes[0,0].set_ylabel('ms'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3, axis='y')
for bar in list(b1)+list(b2):
    axes[0,0].annotate(f'{bar.get_height():.1f}',
                       (bar.get_x()+bar.get_width()/2, bar.get_height()),
                       xytext=(0,3), textcoords='offset points', ha='center', fontsize=9)

axes[0,1].hist(df['TcpRtt_ms'].clip(0,200), bins=60, alpha=0.6,
               label='Avant', color='#A32D2D', edgecolor='none')
axes[0,1].hist(df_opt['TcpRtt_ms'].clip(0,200), bins=60, alpha=0.6,
               label='Après', color='#3B6D11', edgecolor='none')
axes[0,1].set_title('Distribution latence avant/après')
axes[0,1].set_xlabel('TcpRtt_ms'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

slice_label_means = df.groupby(['slice','Label'])['TcpRtt_ms'].mean().unstack()
slice_label_means.plot(kind='bar', ax=axes[0,2], color=['#185FA5','#A32D2D'],
                        alpha=0.85, edgecolor='none')
axes[0,2].set_title('Latence moy. par slice × label')
axes[0,2].set_xlabel(''); axes[0,2].tick_params(axis='x', rotation=0)
axes[0,2].legend(['Benign','Malicious']); axes[0,2].grid(True, alpha=0.3, axis='y')

pl_av = [df[df.slice==s]['pLoss'].mean() for s in slices_list]
pl_ap = [df_opt[df_opt.slice==s]['pLoss'].mean() for s in slices_list]
b3 = axes[1,0].bar(x-w/2, pl_av, w, label='Avant', color='#A32D2D', alpha=0.85, edgecolor='none')
b4 = axes[1,0].bar(x+w/2, pl_ap, w, label='Après', color='#3B6D11', alpha=0.85, edgecolor='none')
axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(slices_list)
axes[1,0].set_title('Taux perte moy. avant/après (%)')
axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3, axis='y')

causes = {'Latence > SLA':df[(df.slice=='URLLC')&(df.TcpRtt_ms>1)].shape[0],
          'Pertes élevées':df[df.pLoss>1].shape[0],
          'Charge réseau':df[df.Load>df.Load.quantile(0.9)].shape[0],
          'Flux malveillant':df[df.anomalie==1].shape[0],
          'SynAck lent':df[df.SynAck>df.SynAck.quantile(0.9)].shape[0]}
axes[1,1].barh(list(causes.keys()), list(causes.values()),
               color=['#A32D2D','#BA7517','#185FA5','#A32D2D','#BA7517'], alpha=0.85)
axes[1,1].set_title("Causes d'anomalies détectées")
axes[1,1].set_xlabel("Nb flux concernés"); axes[1,1].grid(True, alpha=0.3, axis='x')

axes[1,2].axis('off')
t_data = [['Métrique','Avant','Après','Gain'],
           ['Lat. glob. moy.', f'{av_global:.1f}ms', f'{ap_global:.1f}ms',
            f'−{(av_global-ap_global)/av_global*100:.0f}%'],
           ['Lat. URLLC',
            f"{df[df.slice=='URLLC']['TcpRtt_ms'].mean():.2f}ms",
            f"{df_opt[df_opt.slice=='URLLC']['TcpRtt_ms'].mean():.2f}ms",
            f"−{(df[df.slice=='URLLC']['TcpRtt_ms'].mean()-df_opt[df_opt.slice=='URLLC']['TcpRtt_ms'].mean())/df[df.slice=='URLLC']['TcpRtt_ms'].mean()*100:.0f}%"],
           ['pLoss moy.',
            f"{df['pLoss'].mean():.3f}%",
            f"{df_opt['pLoss'].mean():.3f}%",
            f"−{(df['pLoss'].mean()-df_opt['pLoss'].mean())/df['pLoss'].mean()*100:.0f}%"]]
tbl = axes[1,2].table(cellText=t_data[1:], colLabels=t_data[0],
                       cellLoc='center', loc='center', bbox=[0,0,1,1])
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
for (r,c), cell in tbl.get_celld().items():
    if r==0: cell.set_facecolor('#0C447C'); cell.set_text_props(color='white',fontweight='bold')
    elif c==3 and r>0: cell.set_facecolor('#EAF3DE')
axes[1,2].set_title('Bilan optimisation', fontsize=11)

plt.tight_layout()
plt.savefig('results/10_optimisation.png', dpi=150, bbox_inches='tight')
plt.close(); print("\n results/10_optimisation.png")
print(" Étape 5 terminée.")
