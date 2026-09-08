import pandas as pd
import matplotlib.pyplot as plt

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid') 

percorso_csv = "Distanze L2 - Sheet4.csv" 
df = pd.read_csv(percorso_csv)

df.columns = df.columns.str.strip()

modelli = df['MODELLO'].dropna().unique()

attacchi = ['AA', 'PI', 'AP', 'Steering']

colori_metriche = {
    'Overshooting': '#e63946',  # Rosso
    'Forza': '#f4a261',         # Arancione
    'Confine': '#2a9d8f'        # Verde/Acqua
}

print(f"Inizio generazione grafici per {len(modelli)} modelli...")

# 4. GENERAZIONE GRAFICI
for modello in modelli:
    df_modello = df[df['MODELLO'] == modello]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Andamento Distanze L2 - {modello}', fontsize=18, fontweight='bold', y=0.98)
    axes = axes.flatten()
    
    for i, attacco in enumerate(attacchi):
        ax = axes[i]
        if 'Baseline_Vanilla' in df_modello.columns:
            ax.plot(df_modello['LAYER'], df_modello['Baseline_Vanilla'], 
                    label='Baseline Vanilla', color='black', linewidth=2.5, linestyle='--')
        
        for metrica, colore in colori_metriche.items():
            col_name = f"{attacco}_{metrica}"
            
            if col_name in df_modello.columns:
                ax.plot(df_modello['LAYER'], df_modello[col_name], 
                        label=metrica, color=colore, linewidth=2, marker='o', markersize=5)
                
        ax.set_title(f'Attacco: {attacco}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Distanza L2', fontsize=12)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.5)
        
    # Ottimizza gli spazi tra i grafici
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    nome_file_pulito = str(modello).replace('/', '_').replace('\\', '_')
    
    nome_salvataggio = f"anda/{nome_file_pulito}_L2_Trends.png"
    
    plt.savefig(nome_salvataggio, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" -> Salvato: {nome_salvataggio}")

print("Generazione completata")
