import pandas as pd
import os

print("Avvio riparazione CSV...")

# 1. Carichiamo il dataset dei True Positives (che contiene il codice pulito)
percorso_tp = "CSV tesi/Dataset/dataset_TP.csv"
if not os.path.exists(percorso_tp):
    print(f"Errore: Non trovo {percorso_tp}")
    exit()

df_tp = pd.read_csv(percorso_tp)
os.makedirs("CSV tesi/Fixed", exist_ok=True)

# 2. La lista dei file degli attacchi da sistemare
file_attacchi = [
    "CSV tesi/risultati_attacco_steering.csv",
]

for file_csv in file_attacchi:
    if os.path.exists(file_csv):
        df_attacco = pd.read_csv(file_csv)
        
        # id_snippet all'indice del dataset_TP e copiamo il codice
        df_attacco['codice_originale'] = df_attacco['id_snippet'].map(df_tp['codice'])
        
        # Riordiniamo le colonne 
        cols = df_attacco.columns.tolist()
        if 'codice_originale' in cols and 'codice_perturbato' in cols:
            cols.insert(cols.index('codice_perturbato'), cols.pop(cols.index('codice_originale')))
            df_attacco = df_attacco[cols]

        nome_file = os.path.basename(file_csv)
        
        file_output = f"CSV tesi/Fixed/{nome_file}"
        
        # Sovrascriviamo il file
        df_attacco.to_csv(file_output, index=False)
        print(f"[+] Sistemato con successo: {file_csv}")
    else:
        print(f"[-] File non trovato (verrà saltato): {file_csv}")

print("\nRiparazione completata! Sei pronto per l'Isomorfismo.")