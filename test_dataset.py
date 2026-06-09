

import pandas as pd


df_aa = pd.read_csv("CSV tesi/Split_Dataset_succ/advanced_solo_successi.csv")
df_ap = pd.read_csv("CSV tesi/Split_Dataset_succ/ap_solo_successi.csv")
df_pj = pd.read_csv("CSV tesi/Split_Dataset_succ/PJ_solo_successi.csv")
df_ss = pd.read_csv("CSV tesi/Split_Dataset_succ/steering_solo_successi.csv")

df_aa = df_aa.rename(columns={'target_predetto_adv': 'predizione_attacco'})
df_ap = df_ap.rename(columns={'target_predetto_ap': 'predizione_attacco'}) 
df_pj = df_pj.rename(columns={'target_predetto_pj': 'predizione_attacco'})
df_ss = df_ss.rename(columns={'predizione_post_steering': 'predizione_attacco'})

df_aa['tipo_attacco'] = 'Advanced'
df_ap['tipo_attacco'] = 'Adversarial_Perturbation'
df_pj['tipo_attacco'] = 'Prompt_Injection'
df_ss['tipo_attacco'] = 'Steering'

# 4. Concatena i DataFrame
# ignore_index=True resetta gli indici di riga da 0 a N, evitando duplicati (es. due righe con indice 0)
df_combinato = pd.concat([df_aa, df_ap, df_pj, df_ss], ignore_index=True)

# 5. Salva il nuovo CSV
percorso_salvataggio = "CSV tesi/Split_Dataset_succ/tutti_attacchi_successi.csv"
df_combinato.to_csv(percorso_salvataggio, index=False)

print("\n--- UNIONE COMPLETATA ---")
print(f"Righe Advanced: {len(df_aa)}")
print(f"Righe Adversarial Perturbation: {len(df_ap)}")
print(f"Righe Prompt Injection: {len(df_pj)}")
print(f"Totale righe nel dataset combinato: {len(df_combinato)}")
print(f"File salvato in: {percorso_salvataggio}")