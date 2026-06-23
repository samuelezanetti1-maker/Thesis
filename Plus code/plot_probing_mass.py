import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURAZIONE ---
# Questo è il file generato da mass_diagnostic.py
file_csv = "CSV tesi/Diagnostica/risultati_probing_massivo.csv"
cartella_output = "Grafici_Tesi/cartella_output_massivo"

# Creiamo la cartella ESATTA in cui andranno salvati i grafici
os.makedirs(cartella_output, exist_ok=True)


if not os.path.exists(file_csv):
    print(f"[!] Errore: File {file_csv} non trovato. Devi prima eseguire mass_diagnostic.py!")
    exit()

# Caricamento dati
df = pd.read_csv(file_csv)

# Calcoliamo la media delle probabilità per ogni modello e layer
# Stiamo mediando su tutti i 50 snippet per avere una curva pulita e statisticamente solida
df_medie = df.groupby(['modello', 'layer'])[['probabilita_vergine', 'probabilita_attaccata']].mean().reset_index()

# Generiamo un grafico per ogni modello presente nel CSV
modelli = df_medie['modello'].unique()

print(f"Generazione grafici Radar per {len(modelli)} modelli...")

for modello in modelli:
    df_modello = df_medie[df_medie['modello'] == modello]
    
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # 1. Linea Modello Vergine (La Baseline di Sicurezza)
    plt.plot(df_modello['layer'], df_modello['probabilita_vergine'], 
             label='Modello Vergine (Lettura Naturale)', 
             color='blue', marker='o', linewidth=2.5)
             
    # 2. Linea Modello Attaccato (L'Infezione di Steering)
    plt.plot(df_modello['layer'], df_modello['probabilita_attaccata'], 
             label='Modello Attaccato (Infezione Semantica)', 
             color='red', marker='X', linestyle='--', linewidth=2.5)
             
    # Estetica del grafico
    plt.title(f"Radar Semantico: Rilevamento Attacco in Tempo Reale\nModello: {modello.split('/')[-1]}", fontsize=15, fontweight='bold')
    plt.xlabel("Indice del Layer", fontsize=13)
    plt.ylabel("Probabilità 'Vulnerabile' rilevata dal Prober (%)", fontsize=13)
    
    # Fissiamo l'asse Y da 0 a 100 per mantenere le proporzioni
    plt.ylim(-5, 105) 
    
    # Mostriamo un'etichetta sull'asse X ogni 2 layer per non affollare la vista
    plt.xticks(df_modello['layer'][::2]) 
    
    # Aggiungiamo una soglia visiva di allarme (es. al 50%)
    plt.axhline(y=50, color='orange', linestyle=':', linewidth=2, label="Soglia di Trigger Sentinella (50%)")
    
    # Mettiamo la legenda in una posizione comoda
    plt.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    
    # Salvataggio
    nome_file_pulito = modello.replace("/", "_")
    percorso_salvataggio = f"{cartella_output}/Radar_Sentinella_{nome_file_pulito}.png"
    plt.savefig(percorso_salvataggio, dpi=300)
    plt.close()
    
    print(f" -> Salvato: {percorso_salvataggio}")

print("\n[+] Tutti i grafici diagnostici sono stati generati!")