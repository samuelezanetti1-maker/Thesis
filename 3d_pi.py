import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from config import models_config

sns.set_theme(style="whitegrid", context="talk")
os.makedirs("L2_COMPLETI", exist_ok=True)

def plot_tetraedro_3d_ancorato(distanze, model_name, percorso_salvataggio):
    d_base, d_S_AA_to_S, d_S_AA_to_V, d_V_AA_to_S, d_V_AA_to_V, d_confine = distanze

    # =========================================================
    # 1. TRILATERAZIONE EUCLIDEA ESATTA
    # =========================================================
    p1 = np.array([0.0, 0.0, 0.0]) # Vulnerabile
    p0 = np.array([d_base, 0.0, 0.0]) # Sicuro

    x2 = (d_S_AA_to_V**2 + d_base**2 - d_S_AA_to_S**2) / (2 * d_base)
    y2 = np.sqrt(max(0, d_S_AA_to_V**2 - x2**2))
    p2 = np.array([x2, y2, 0.0])

    x3 = (d_V_AA_to_V**2 + d_base**2 - d_V_AA_to_S**2) / (2 * d_base)
    if y2 != 0:
        y3 = (d_V_AA_to_V**2 + d_S_AA_to_V**2 - d_confine**2 - 2 * x3 * x2) / (2 * y2)
    else:
        y3 = 0.0
    z3 = np.sqrt(max(0, d_V_AA_to_V**2 - x3**2 - y3**2))
    p3 = np.array([x3, y3, z3])

    coords = np.array([p0, p1, p2, p3])

    # =========================================================
    # 2. SETUP DEL GRAFICO 3D
    # =========================================================
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    labels = ['Sicuro (Base)', 'Vulnerabile (Base)', 'Attacco (Riuscito)', 'Attacco (Fallito)']
    colori = ['blue', 'red', 'purple', 'orange']
    markers = ['o', 'o', 'o', 'o']
    centroide = np.mean(coords, axis=0)

    # =========================================================
    # 3. PROIEZIONI DI PROFONDITÀ (Drop Lines & Ombre) <-- LA NOVITÀ!
    # =========================================================
    for i, color in zip([2, 3], ['purple', 'orange']):
        x, y, z = coords[i]
        
        # Disegno la "gamba" verticale che cade sul pavimento (Piano X-Y)
        ax.plot([x, x], [y, y], [0, z], color=color, linestyle=':', alpha=0.6, linewidth=1.5)
        
        # Disegno l'ombra del punto sul pavimento
        ax.scatter(x, y, 0, c=color, marker=markers[i], s=50, alpha=0.3)
        
        # Disegno le linee di guida sul pavimento per X e Y
        ax.plot([x, x], [0, y], [0, 0], color='gray', linestyle=':', alpha=0.4)
        ax.plot([0, x], [y, y], [0, 0], color='gray', linestyle=':', alpha=0.4)

   # =========================================================
    # 4. DISEGNO DEI PUNTI E DELLE ETICHETTE (Effetto "Bandierina")
    # =========================================================
    for i in range(4):
        # Disegno il punto principale
        ax.scatter(coords[i, 0], coords[i, 1], coords[i, 2], 
                   c=colori[i], marker=markers[i], s=200, label=labels[i], zorder=10)
        
        # Alziamo il testo verticalmente (lungo l'asse Z)
        altezza_asta = d_base * 0.35  # Modifica questo per fare l'asta più o meno lunga
        x_testo = coords[i, 0]
        y_testo = coords[i, 1]
        z_testo = coords[i, 2] + altezza_asta

        # Disegno l'asta (il filo che collega il punto alla scatola)
        ax.plot([coords[i, 0], x_testo], 
                [coords[i, 1], y_testo], 
                [coords[i, 2], z_testo], 
                color='black', linestyle='-', linewidth=1.5, alpha=0.5, zorder=5)

        # Aggiungo un bordino leggero nero (ec='black') alla scatola per farla staccare dallo sfondo
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.95)
        
        # va='bottom' è vitale: la scatola di testo viene appoggiata SOPRA l'asta, mai sopra il punto
        ax.text(x_testo, y_testo, z_testo, labels[i], 
                fontsize=11, fontweight='bold', zorder=20, 
                ha='center', va='bottom', bbox=bbox_props)

    # =========================================================
    # 5. DISEGNO DEGLI SPIGOLI
    # =========================================================
    def draw_line(p1, p2, color, style='-'):
        ax.plot([coords[p1, 0], coords[p2, 0]],
                [coords[p1, 1], coords[p2, 1]],
                [coords[p1, 2], coords[p2, 2]], 
                color=color, linestyle=style, alpha=0.5, linewidth=2, zorder=5)

    draw_line(0, 1, 'gray', '--')  
    draw_line(1, 2, 'purple', '-') 
    draw_line(0, 2, 'purple', ':') 
    draw_line(1, 3, 'orange', '-') 
    draw_line(0, 3, 'orange', ':') 
    draw_line(2, 3, 'green', '--') 

    # =========================================================
    # 6. ESTETICA E CAMERA
    # =========================================================
    ax.set_title(f"Topologia Spaziale - Prompt Injection \n{model_name}", fontweight='bold', pad=20)
    
    ax.set_xlabel("", labelpad=15, fontweight='bold')
    ax.set_ylabel("", labelpad=15)
    ax.set_zlabel("", labelpad=15)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # RUOTO LA TELECAMERA PER UNA PROSPETTIVA MIGLIORE (Elevazione 20, Angolo 40)
    ax.view_init(elev=20, azim=40)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    
    plt.savefig(percorso_salvataggio, dpi=300, bbox_inches='tight', format='png')
    plt.close()


# =========================================================
# CARICAMENTO DATI ED ESECUZIONE
# =========================================================
df = pd.read_csv("Distanze_L2_values.csv")

layer_per_modello = {
    "Qwen/Qwen2.5-7B-Instruct": 26,
    "Qwen/Qwen2.5-Coder-7B-Instruct": 26,
    "meta-llama/Llama-3.1-8B-Instruct": 30,
    "codellama/CodeLlama-7b-Instruct-hf": 30,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 26,
    "deepseek-ai/deepseek-coder-6.7b-instruct": 30
}

for model_name, config in models_config.items():
    print(f"\nElaborazione modello: {model_name}")
    layer_locus = layer_per_modello.get(model_name)

    df_filtrato = df[(df['MODELLO'] == model_name) & (df['LAYER'] == layer_locus)]
    if len(df_filtrato) == 0:
        continue

    # Estrazione dei valori scalari esatti
    distanze_array = [
        df_filtrato["Baseline_Vanilla (S_base - V_base)"].values[0],
        df_filtrato["PI_Overshooting (S_PI - S_base)"].values[0],
        df_filtrato["PI_Forza (S_PI - V_base)"].values[0],
        df_filtrato["PI (S_base - V_PI)"].values[0],
        df_filtrato["PI (V_base - V_PI)"].values[0],
        df_filtrato["PI_Confine (S_PI - V_PI)"].values[0]
    ]

    nome_file_safe = model_name.replace('/', '_')
    percorso = f"L2_COMPLETI/3D/last_layer/PI_Topologia3D_{nome_file_safe}_{layer_locus}.png"
    
    plot_tetraedro_3d_ancorato(distanze_array, model_name, percorso)
    print(f" -> Grafico salvato in: {percorso}")


      
    