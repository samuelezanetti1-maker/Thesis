import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import torch
import gc


from transformers import AutoModelForCausalLM, AutoTokenizer
from config import models_config, df_globale
from utils import traduci_target_reale, evaluate_response, is_true_positive, is_true_negative

risultati_baseline = []
df = df_globale.copy()

for model_name, config in models_config.items():
    print("\n" + "="*60)
    print(f"INIZIO TEST CON IL MODELLO: {model_name}")
    print("="*60)

    nome_modello_pulito = model_name.replace("/", "_")
    percorso_txt = f"txt_tesi/Log_Risposte_Baseline_{nome_modello_pulito}.txt"

    with open(percorso_txt, "w", encoding="utf-8") as f_log:
        f_log.write(f"=== LOG RISPOSTE GREZZE: {model_name} ===\n\n")

    # 1. Caricamento Modello e Tokenizer
    try:
        # trust_remote_code=True è necessario per modelli particolari come Phi
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=config.get("quantization_config", None),
            device_map="auto",
            low_cpu_mem_usage=True,
            dtype=config.get("dtype", torch.float16)
        )
    except Exception as e:
        print(f"Errore nel caricamento di {model_name}: {e}")
        continue # Se un modello fallisce, salta al prossimo senza far crashare tutto

    
    # 2. Ciclo sui dati 
    for index, row in df.iterrows():
        codice = row['func']
        target = row['target']
        
        # Estraggo solo l'ultima parte del nome per stamparlo più corto
        nome_corto = model_name.split('/')[-1]
        print(f"\nAnalisi Snippet {index+1}/{len(df)} | Modello: {nome_corto}")
        
        prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
        messages = [
            {"role": "system", "content": "You are a cybersecurity expert. Your task is to find vulnerabilities in the source code."},
            {"role": "user", "content": prompt}
        ]
        
        # Applichiamo il template specifico di ogni modello
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        token_limit = 1500 if "deepseek" in model_name.lower() else 250
        print("Generazione risposta in corso...")
        generated_ids_base = model.generate(
            **model_inputs, 
            max_new_tokens=token_limit, 
            do_sample=False
        )
        output_ids_base = generated_ids_base[0][len(model_inputs.input_ids[0]):]
        risposta_base = tokenizer.decode(output_ids_base, skip_special_tokens=True)
        

        #per txt log
        with open(percorso_txt, "a", encoding="utf-8") as f_log:
            f_log.write(f"Snippet ID {index}:\n")
            f_log.write(f"{risposta_base}\n")
            f_log.write("-" * 50 + "\n\n")

        risposta_pulita = risposta_base.replace('\n', ' ')
        print(f" |RISPOSTA|: {risposta_pulita[:120]}...")

        risultati_baseline.append({
            "id_snippet": index,
            "modello": model_name,
            "codice": codice,
            "target_reale": target,
            "risposta_baseline": risposta_base
        })

    # 3. Salvataggio risultati
    nome_file_safe = model_name.replace('/', '_')
    df_risultati = pd.DataFrame(risultati_baseline)
    df_risultati.to_csv(f"CSV tesi/risultati_baseline.csv", index=False)
    print(f"\n Risultati di {nome_corto} salvati in 'risultati_baseline_{nome_file_safe}.csv'")

    # 4. Pulizia memoria
    try:
        del model
        del tokenizer
        del model_inputs
        del generated_ids_base
    except NameError:
        pass 

    gc.collect() # Invoca il Garbage Collector di Python
    torch.cuda.empty_cache() # Svuota fisicamente la memoria video della scheda grafica


file_csv = "CSV tesi/risultati_baseline.csv"
print(f" Lettura del file: {file_csv}")



try:
    df = pd.read_csv(file_csv)

    # Applichiamo le traduzioni
    df['target_predetto'] = df['risposta_baseline'].apply(evaluate_response)
    df['target_vero'] = df['target_reale'].apply(traduci_target_reale)

    # Filtriamo via le risposte ambigue per la matrice di confusione
    df_validi = df[df['target_predetto'] != "Non Classificato"]
    
    print("\n" + "="*50)
    print(" ANALISI METRICHE E MATRICE DI CONFUSIONE")
    print("="*50)

    # --- CALCOLO PER SINGOLO MODELLO ---
    for modello in df['modello'].unique():
        df_mod = df_validi[df_validi['modello'] == modello]
        
        if len(df_mod) == 0:
            print(f"\nNessuna risposta valida per {modello}")
            continue
            
        totale_mod = len(df_mod)
        
        # Calcolo dei 4 quadranti della Confusion Matrix
        # TP: Vero Positivo (Vulnerabile predetto Vulnerabile)
        tp = len(df_mod[(df_mod['target_vero'] == 'Vulnerabile') & (df_mod['target_predetto'] == 'Vulnerabile')])
        
        # FN: Falso Negativo (Vulnerabile predetto Sicuro) -> IL PIÙ PERICOLOSO!
        fn = len(df_mod[(df_mod['target_vero'] == 'Vulnerabile') & (df_mod['target_predetto'] == 'Sicuro')])
        
        # FP: Falso Positivo (Sicuro predetto Vulnerabile)
        fp = len(df_mod[(df_mod['target_vero'] == 'Sicuro') & (df_mod['target_predetto'] == 'Vulnerabile')])
        
        # TN: Vero Negativo (Sicuro predetto Sicuro)
        tn = len(df_mod[(df_mod['target_vero'] == 'Sicuro') & (df_mod['target_predetto'] == 'Sicuro')])

        # Calcolo percentuali
        perc_tp = (tp / totale_mod) * 100
        perc_fn = (fn / totale_mod) * 100
        perc_fp = (fp / totale_mod) * 100
        perc_tn = (tn / totale_mod) * 100
        
        accuracy = ((tp + tn) / totale_mod) * 100
        
        nome_corto = modello.split('/')[-1]
        print(f"\n MODELLO: {nome_corto}")
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   --- Matrice di Confusione ---")
        print(f"   Vero Vulnerabile (TP) : {perc_tp:.1f}% ({tp}) - Vulnerabile predetto Vulnerabile")
        print(f"   Falso Negativo   (FN) : {perc_fn:.1f}% ({fn}) -  Vulnerabile predetto Sicuro")
        print(f"   Falso Positivo   (FP) : {perc_fp:.1f}% ({fp}) - Sicuro predetto Vulnerabile")
        print(f"   Vero Sicuro      (TN) : {perc_tn:.1f}% ({tn}) - Sicuro predetto Sicuro")

    # --- CREAZIONE CSV PULITO ---
    df_finale = df[['modello', 'target_vero', 'target_predetto', 'codice']]
    df_finale.to_csv("CSV tesi/risultati_semplificati.csv", index=False)
    
    # Check dei non classificati
    non_capiti = len(df[df['target_predetto'] == "Non Classificato"])
    if non_capiti > 0:
        print(f"\n Attenzione: {non_capiti} risposte totali su {len(df)} non sono state classificate (manca 'FINAL_VERDICT').")

except FileNotFoundError:
    print(f" Errore: Il file {file_csv} non è stato trovato nella cartella.")

### Divisione dataset TP
risultati_baseline_semplificati = "CSV tesi/risultati_semplificati.csv"

df = pd.read_csv(risultati_baseline_semplificati)
df_modello_TP = is_true_positive(df)
df_modello_TP.to_csv("CSV tesi/dataset_TP.csv", index=False)
df_modello_TN = is_true_negative(df)
df_modello_TN.to_csv("CSV tesi/dataset_TN.csv", index=False)
df_modello_TRUE = pd.concat([df_modello_TP, df_modello_TN])
df_modello_TRUE.to_csv("CSV tesi/dataset_TRUE.csv", index=False)