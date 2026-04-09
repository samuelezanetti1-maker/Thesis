import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# --- CONFIGURAZIONE ---
# Metti il nome esatto del modello (come salvato nei file .pkl)
nome_modello = "Qwen_Qwen2.5-7B-Instruct" 
cartella_attivazioni = "attivazioni_totali"
cartella_output = "Grafici_Tesi"

os.makedirs(cartella_output, exist_ok=True)

# Cerchiamo tutti i file .pkl
pattern_ricerca = f"{cartella_attivazioni}/*{nome_modello}*layer_*.pkl"
file_trovati = glob.glob(pattern_ricerca)

if not file_trovati:
    print(f"Nessun file trovato per {nome_modello}.")
    exit()

# ==============================================================================
# ESPERIMENTO 1: FOTOGRAFIA DI UN SINGOLO LAYER (es. Il Locus Causale)
# ==============================================================================
layer_bersaglio = 18  # Scegli il layer che ha performato meglio nello sweep!

file_layer = f"{cartella_attivazioni}/vettori_{nome_modello}_layer_{layer_bersaglio}.pkl"

if os.path.exists(file_layer):
    print(f" Generazione grafico Spazio Latente per il Layer {layer_bersaglio}...")
    df = pd.read_pickle(file_layer)
    df = df.dropna(subset=['vettore_attivazione', 'target_vero'])
    
    X = np.stack(df['vettore_attivazione'].values)
    y = df['target_vero'].values
    
    # Applichiamo la PCA per ridurre da 4096 a 2 dimensioni
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Dividiamo i punti in base alla classe
    idx_sicuri = (y == 'Sicuro')
    idx_vulnerabili = (y == 'Vulnerabile')
    
    X_sicuri = X_pca[idx_sicuri]
    X_vulnerabili = X_pca[idx_vulnerabili]
    
    # Calcoliamo i centroidi in 2D
    centroide_sicuro = np.mean(X_sicuri, axis=0)
    centroide_vulnerabile = np.mean(X_vulnerabili, axis=0)
    
    # --- PLOT ---
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    
    # Disegniamo le nuvole di punti
    plt.scatter(X_sicuri[:, 0], X_sicuri[:, 1], c='blue', alpha=0.3, label='Codici Sicuri', edgecolors='w')
    plt.scatter(X_vulnerabili[:, 0], X_vulnerabili[:, 1], c='red', alpha=0.3, label='Codici Vulnerabili', edgecolors='w')
    
    # Disegniamo i Centroidi
    plt.scatter(*centroide_sicuro, c='darkblue', marker='*', s=400, edgecolors='black', label='Centroide Sicuro')
    plt.scatter(*centroide_vulnerabile, c='darkred', marker='*', s=400, edgecolors='black', label='Centroide Vulnerabile')
    
    # DISEGNIAMO IL VETTORE DI STEERING (La freccia)
    plt.annotate('', xy=centroide_vulnerabile, xytext=centroide_sicuro,
                 arrowprops=dict(facecolor='black', width=3, headwidth=10, shrink=0))
    
    plt.title(f"Spazio Latente (PCA) - Layer {layer_bersaglio}\nModello: {nome_modello}", fontsize=14, fontweight='bold')
    plt.xlabel(f"Componente Principale 1 ({pca.explained_variance_ratio_[0]*100:.1f}% invarianza)")
    plt.ylabel(f"Componente Principale 2 ({pca.explained_variance_ratio_[1]*100:.1f}% invarianza)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cartella_output}/Spazio_Latente_L{layer_bersaglio}_{nome_modello}.png", dpi=300)
    print("Grafico del layer salvato!")

# ==============================================================================
# ESPERIMENTO 2: TRAIETTORIA DEI CENTROIDI SU TUTTI I LAYER
# ==============================================================================
print("\n Generazione grafico della Traiettoria su tutti i layer...")

tutti_i_dati = []
layer_indices = []

# Carichiamo tutti i file per trovare uno spazio PCA comune
for file_pkl in sorted(file_trovati, key=lambda x: int(re.search(r'layer_(\d+)', x).group(1))):
    layer_idx = int(re.search(r'layer_(\d+)', file_pkl).group(1))
    df = pd.read_pickle(file_pkl).dropna()
    
    X = np.stack(df['vettore_attivazione'].values)
    y = df['target_vero'].values
    
    tutti_i_dati.append((layer_idx, X, y))
    layer_indices.append(layer_idx)

# Concateniamo tutto per addestrare una singola PCA globale
X_globale = np.concatenate([d[1] for d in tutti_i_dati])
pca_globale = PCA(n_components=2)
pca_globale.fit(X_globale) # Addestrata su tutti i layer!

traj_sicuri = []
traj_vulnerabili = []

for layer_idx, X, y in tutti_i_dati:
    X_pca = pca_globale.transform(X)
    
    c_sicuro = np.mean(X_pca[y == 'Sicuro'], axis=0)
    c_vulnerabile = np.mean(X_pca[y == 'Vulnerabile'], axis=0)
    
    traj_sicuri.append(c_sicuro)
    traj_vulnerabili.append(c_vulnerabile)

traj_sicuri = np.array(traj_sicuri)
traj_vulnerabili = np.array(traj_vulnerabili)

# --- PLOT TRAIETTORIA ---
plt.figure(figsize=(12, 8))
sns.set_theme(style="darkgrid")

# Disegniamo i percorsi
plt.plot(traj_sicuri[:, 0], traj_sicuri[:, 1], marker='o', color='blue', label='Evoluzione Concetto: Sicuro', linewidth=2, alpha=0.7)
plt.plot(traj_vulnerabili[:, 0], traj_vulnerabili[:, 1], marker='s', color='red', label='Evoluzione Concetto: Vulnerabile', linewidth=2, alpha=0.7)

# Annotiamo i numeri dei layer ogni 5 step per non fare confusione
for i, layer_idx in enumerate(layer_indices):
    if layer_idx % 5 == 0 or layer_idx == max(layer_indices):
        plt.annotate(f"L{layer_idx}", (traj_sicuri[i, 0], traj_sicuri[i, 1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='darkblue')
        plt.annotate(f"L{layer_idx}", (traj_vulnerabili[i, 0], traj_vulnerabili[i, 1]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='darkred')

# Disegniamo delle linee grigie leggere che collegano le due idee ad ogni layer (le norme L2!)
for i in range(len(layer_indices)):
    plt.plot([traj_sicuri[i, 0], traj_vulnerabili[i, 0]], 
             [traj_sicuri[i, 1], traj_vulnerabili[i, 1]], 
             color='gray', linestyle=':', alpha=0.3)

plt.title(f"Traiettoria dei Centroidi nei Layer (PCA Globale)\nModello: {nome_modello}", fontsize=14, fontweight='bold')
plt.xlabel("Dimensione Latente Principale 1")
plt.ylabel("Dimensione Latente Principale 2")
plt.legend()
plt.tight_layout()
plt.savefig(f"{cartella_output}/Traiettoria_TuttiLayer_{nome_modello}.png", dpi=300)
print(f"Grafico traiettoria salvato in {cartella_output}!")