import pandas as pd
import matplotlib.pyplot as plt

# 1. IMPOSTAZIONI GRAFICHE
# Usa uno stile pulito e accademico (puoi usare 'ggplot' o 'seaborn-whitegrid' se preferisci)
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid') # Fallback per versioni più vecchie di matplotlib

# 2. CARICAMENTO DATI
percorso_csv = "Distanze L2 - Sheet4.csv" # Sostituisci con il nome reale del tuo file
df = pd.read_csv(percorso_csv)

# Pulizia dei nomi delle colonne (rimuove eventuali spazi vuoti accidentali a inizio/fine)
df.columns = df.columns.str.strip()

# 3. DEFINIZIONE VARIABILI
# Estrai la lista di tutti i modelli unici presenti nel CSV
modelli = df['MODELLO'].dropna().unique()

# Definizione dei 4 attacchi da plottare
attacchi = ['AA', 'PI', 'AP', 'Steering']

# Palette di colori fissa per mantenere coerenza visiva tra tutti i grafici
colori_metriche = {
    'Overshooting': '#e63946',  # Rosso
    'Forza': '#f4a261',         # Arancione
    'Confine': '#2a9d8f'        # Verde/Acqua
}

print(f"Inizio generazione grafici per {len(modelli)} modelli...")

# 4. GENERAZIONE GRAFICI
for modello in modelli:
    # Filtra il dataframe solo per il modello corrente
    df_modello = df[df['MODELLO'] == modello]
    
    # Crea una figura con una griglia 2x2
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Titolo principale della figura
    fig.suptitle(f'Andamento Distanze L2 - {modello}', fontsize=18, fontweight='bold', y=0.98)
    
    # Appiattisci l'array degli assi per iterare facilmente da 0 a 3
    axes = axes.flatten()
    
    for i, attacco in enumerate(attacchi):
        ax = axes[i]
        
        # A) Plotta la Baseline (Linea nera tratteggiata)
        if 'Baseline_Vanilla' in df_modello.columns:
            ax.plot(df_modello['LAYER'], df_modello['Baseline_Vanilla'], 
                    label='Baseline Vanilla', color='black', linewidth=2.5, linestyle='--')
        
        # B) Plotta le tre metriche per l'attacco corrente
        for metrica, colore in colori_metriche.items():
            col_name = f"{attacco}_{metrica}"
            
            # Controlla se la colonna esiste per quell'attacco
            if col_name in df_modello.columns:
                ax.plot(df_modello['LAYER'], df_modello[col_name], 
                        label=metrica, color=colore, linewidth=2, marker='o', markersize=5)
                
        # C) Formattazione del singolo sotto-grafico
        ax.set_title(f'Attacco: {attacco}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Distanza L2', fontsize=12)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.5)
        
    # Ottimizza gli spazi tra i grafici
    plt.tight_layout()
    # Lascia spazio in alto per il titolo principale
    plt.subplots_adjust(top=0.92)
    
    # D) Salvataggio del file
    # Pulisci il nome del modello per evitare errori nel nome del file (es. togli le barre '/')
    nome_file_pulito = str(modello).replace('/', '_').replace('\\', '_')
    
    nome_salvataggio = f"anda/{nome_file_pulito}_L2_Trends.png"
    
    plt.savefig(nome_salvataggio, dpi=300, bbox_inches='tight')
    plt.close() # Chiudi la figura per liberare memoria
    
    print(f" -> Salvato: {nome_salvataggio}")

print("Generazione completata!")