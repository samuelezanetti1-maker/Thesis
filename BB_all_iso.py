import os
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import models_config
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/scratch_share/bislab/HF_HUB_CACHE/"

# 1. CARICAMENTO DEI 3 CSV DEGLI ATTACCHI
df_adv = pd.read_csv("CSV tesi/Fixed/risultati_attacco_advanced.csv")
df_pi = pd.read_csv("CSV tesi/Fixed/risultati_attacco_PJ.csv")
df_ap = pd.read_csv("CSV tesi/Fixed/risultati_attacco_adversarial_perturbation.csv")

N_CAMPIONI = 200 
risultati_isomorfismo_layer = []

print("Avvio Analisi Isomorfismo Multiplo (AA vs AP vs PI)...\n" + "="*70)

for model_name, config in models_config.items():
    print(f"\nElaborazione modello: {model_name}")
    
    # 2. FILTRI INDIPENDENTI: Prendiamo i successi di ciascun attacco
    df_aa_succ = df_adv[(df_adv['modello'] == model_name) & (df_adv['target_predetto_adv'] == 'Sicuro')].head(N_CAMPIONI)
    df_pi_succ = df_pi[(df_pi['modello'] == model_name) & (df_pi['target_predetto_pj'] == 'Sicuro')].head(N_CAMPIONI)
    df_ap_succ = df_ap[(df_ap['modello'] == model_name) & (df_ap['target_predetto_ap'] == 'Sicuro')].head(N_CAMPIONI)

    if len(df_aa_succ) == 0 or len(df_pi_succ) == 0 or len(df_ap_succ) == 0:
        print(f"  [!] Dati di successo insufficienti per incrociare i 3 attacchi su {model_name}. Salto.")
        continue

    print(f"  -> Campioni in analisi: AA ({len(df_aa_succ)}), PI ({len(df_pi_succ)}), AP ({len(df_ap_succ)})")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                low_cpu_mem_usage=True,
                dtype=torch.float16
            )

        def get_all_hidden_states(codice):
            # Estrazione puramente spaziale (NO fast-forward)
            prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
            messages = [
                {"role": "system", "content": "You are a cybersecurity expert. Your task is to find vulnerabilities in the source code."},
                {"role": "user", "content": prompt}
            ]
            testo_formattato = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([testo_formattato], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            return torch.stack([layer_state[0, -1, :].detach().cpu() for layer_state in outputs.hidden_states])

        # =========================================================
        # 1. ESTRAZIONE E MEDIA ADVANCED ADVERSARIAL (AA)
        # =========================================================
        print("  -> Estrazione vettori AA...")
        shift_list = []
        for _, row in df_aa_succ.iterrows():
            shift_list.append(get_all_hidden_states(row['codice_perturbato']) - get_all_hidden_states(row['codice_originale']))
        vettore_medio_aa = torch.mean(torch.stack(shift_list), dim=0)
        del shift_list; gc.collect() # Libero la RAM immediatamente!

        # =========================================================
        # 2. ESTRAZIONE E MEDIA PROMPT INJECTION (PI)
        # =========================================================
        print("  -> Estrazione vettori PI...")
        shift_list = []
        for _, row in df_pi_succ.iterrows():
            shift_list.append(get_all_hidden_states(row['codice_perturbato']) - get_all_hidden_states(row['codice_originale']))
        vettore_medio_pi = torch.mean(torch.stack(shift_list), dim=0)
        del shift_list; gc.collect()

        # =========================================================
        # 3. ESTRAZIONE E MEDIA ADVERSARIAL PERTURBATION (AP)
        # =========================================================
        print("  -> Estrazione vettori AP...")
        shift_list = []
        for _, row in df_ap_succ.iterrows():
            shift_list.append(get_all_hidden_states(row['codice_perturbato']) - get_all_hidden_states(row['codice_originale']))
        vettore_medio_ap = torch.mean(torch.stack(shift_list), dim=0)
        del shift_list; gc.collect()

        # =========================================================
        # CALCOLO DELLA TRIPLA CROSS-CORRELATION
        # =========================================================
        num_layers = vettore_medio_aa.shape[0]
        print("  -> Calcolo delle 3 Cosine Similarities layer-by-layer...")
        
        for layer_idx in range(num_layers):
            # Casting a float32 per stabilità numerica
            v_aa = vettore_medio_aa[layer_idx].to(torch.float32)
            v_pi = vettore_medio_pi[layer_idx].to(torch.float32)
            v_ap = vettore_medio_ap[layer_idx].to(torch.float32)
            
            # I Tre Incroci
            sim_aa_ap = F.cosine_similarity(v_aa, v_ap, dim=0).item()
            sim_aa_pi = F.cosine_similarity(v_aa, v_pi, dim=0).item()
            sim_ap_pi = F.cosine_similarity(v_ap, v_pi, dim=0).item()
            
            risultati_isomorfismo_layer.append({
                "MODELLO": model_name,
                "LAYER": layer_idx,
                "Cosine_AA_vs_AP": sim_aa_ap,
                "Cosine_AA_vs_PI": sim_aa_pi,
                "Cosine_AP_vs_PI": sim_ap_pi
            })

        del model, tokenizer
        del vettore_medio_aa, vettore_medio_pi, vettore_medio_ap
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Errore su {model_name}: {e}")
        continue

# 3. SALVATAGGIO FINALE
os.makedirs("CSV tesi/Cross_Sweep", exist_ok=True)
percorso_csv = "CSV tesi/Cross_Sweep/Cross_Correlation_Massiva_Tutti_I_Layer.csv"
df_finale = pd.DataFrame(risultati_isomorfismo_layer)
df_finale.to_csv(percorso_csv, index=False)

print("\n" + "="*70)
print(f" Elaborazione finita! Dati salvati in: {percorso_csv}")