import pandas as pd
import glob

# Trova tutti i file CSV che hai generato per Logit Lens e Isomorfismo
file_da_pulire = [
    #"CSV tesi/Logit_Lens/Risultati_Sweep_Steering.csv",
    #"CSV tesi/Logit_Lens/Risultati_Sweep_Steering_FALLIMENTI.csv",
    "CSV tesi/Logit_Lens/Risultati_Contrastive_All_Layers.csv",
    "CSV tesi/Logit_Lens/Risultati_Contrastive_All_Layers_fallimenti.csv",
    
]

for file_csv in file_da_pulire:
    try:
        df = pd.read_csv(file_csv)
        
        # 1. Rimuoviamo il layer 0 (l'Embedding)
        df_pulito = df[df['LAYER'] != 0].copy()
        
        # 2. Scaliamo gli indici di 1 per allinearli ai blocchi Transformer (da 0 a 27)
        df_pulito['LAYER'] = df_pulito['LAYER'] - 1
        
        # Sovrascriviamo il file CSV
        df_pulito.to_csv(file_csv, index=False)
        print(f"Pulito e riallineato: {file_csv}")
        
    except FileNotFoundError:
        print(f"File non trovato, lo salto: {file_csv}")

print("Tutti i CSV sono ora perfettamente allineati allo Steering (0 -> N-1)!")