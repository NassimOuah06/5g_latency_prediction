import pandas as pd
import os

os.makedirs('results', exist_ok=True)

df = pd.read_csv('data/Global.csv')

df['TcpRtt_ms'] = df['TcpRtt'] * 1000
df['anomalie']  = (df['Label'] == 'Malicious').astype(int)

print("=" * 60)
print("ÉTAPE 1 — DESCRIPTION DU DATASET 5G")
print("=" * 60)
print("\n  Fichier  : Global.csv + URLLC.csv + eMBB.csv + mMTC.csv")
print(f"  Lignes   : {df.shape[0]:,}")
print(f"  Colonnes : {df.shape[1]}")

print("\n Description des colonnes clés ")
desc = {
    'TcpRtt'      : 'Latence TCP aller-retour (s) ← CIBLE (×1000 = ms)',
    'SynAck'      : 'Délai TCP SYN-ACK (s) — fort corrélé à TcpRtt',
    'AckDat'      : 'Délai ACK-Data (s)',
    'Dur'         : 'Durée du flux réseau (s)',
    'TotBytes'    : 'Volume total de données transférées (octets)',
    'TotPkts'     : 'Nombre total de paquets',
    'Load'        : 'Charge réseau (bps)',
    'Loss'        : 'Paquets perdus',
    'pLoss'       : 'Taux de perte (%)',
    'Rate'        : 'Débit moyen (pps)',
    'predicted'   : 'Type de slice 5G : 1:eMBB / 2:mMTC / 3:URLLC',
    'Label'       : 'Benign (normal) / Malicious (anomalie/attaque)',
}
for col, d in desc.items():
    print(f"  {col:<15} : {d}")

print("\n Distribution labels ")
print(f"  Normal   (Benign)   : {(df.anomalie==0).sum():>5}  ({(df.anomalie==0).mean()*100:.1f}%)")
print(f"  Anomalie (Malicious): {(df.anomalie==1).sum():>5}  ({(df.anomalie==1).mean()*100:.1f}%)")

print("\n Distribution par slice 5G ")
print(df.groupby('predicted').agg(
    n=('TcpRtt','count'),
    latence_moy=('TcpRtt_ms', 'mean'),
    latence_max=('TcpRtt_ms', 'max'),
    pct_zeros=('TcpRtt', lambda x: f"{(x==0).mean()*100:.1f}%"),
    pct_malicious=('Label', lambda x: f"{(x=='Malicious').mean()*100:.1f}%")
).round(2).to_string())

print("\n Statistiques TcpRtt_ms (latence) ")
print(df['TcpRtt_ms'].describe().round(3).to_string())

print("\n Valeurs manquantes (colonnes features) ")
features = ["Dur","RunTime","Mean","Sum","Min","Max",
            "TotPkts","SrcPkts","DstPkts","TotBytes","SrcBytes","DstBytes",
            "Load","SrcLoad","DstLoad","Loss","SrcLoss","DstLoss","pLoss",
            "Rate","SrcRate","DstRate","sMeanPktSz","dMeanPktSz","SynAck","AckDat"]
nulls = df[features].isnull().sum()
print(nulls[nulls > 0].to_string() if nulls.any() else "  Aucune valeur manquante dans les features retenues.")

print("\n Étape 1 terminée.")
