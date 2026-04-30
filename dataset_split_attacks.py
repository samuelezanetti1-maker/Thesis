import pandas as pd

# 1. Definisci il nome del file di input e di output
file_input = "CSV tesi/Fixed/risultati_attacco_adversarial_perturbation.csv"
file_output = "CSV tesi/Split_Dataset_succ/ap_solo_successi.csv"

try:
    # 2. Carica il dataset
    df = pd.read_csv(file_input)
    
    # 3. Filtra le righe
    # Uso .str.lower() per essere sicuro di prendere 'Sicuro', 'sicuro' o 'SICURO'
    df_filtrato = df[df['target_predetto_ap'].astype(str).str.lower() == 'sicuro']
    
    # 4. Salva il nuovo dataset (index=False evita di creare una colonna con i numeri di riga)
    df_filtrato.to_csv(file_output, index=False)
    
    # 5. Stampa un resoconto
    print("\n--- OPERAZIONE COMPLETATA ---")
    print(f"Righe totali nel dataset originale: {len(df)}")
    print(f"Attacchi con successo trovati: {len(df_filtrato)}")
    print(f"File salvato correttamente come: {file_output}")

except FileNotFoundError:
    print(f"Errore: Il file '{file_input}' non è stato trovato.")
except KeyError:
    print("Errore: La colonna 'target_predetto_ap' non esiste nel CSV. Controlla il nome esatto.")