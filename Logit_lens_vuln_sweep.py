import os
import pandas as pd
import torch
import gc
import numpy as np
import re
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import models_config

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/scratch_share/bislab/HF_HUB_CACHE/"

df_baseline = pd.read_csv("CSV tesi/Dataset/dataset_TRUE.csv")
os.makedirs("CSV tesi/Logit_Lens", exist_ok=True) 

df_aa = pd.read_csv("CSV tesi/Fixed/risultati_attacco_advanced.csv")
df_pi = pd.read_csv("CSV tesi/Fixed/risultati_attacco_PJ.csv")
df_ap = pd.read_csv("CSV tesi/Fixed/risultati_attacco_adversarial_perturbation.csv")

df_aa_fail = df_aa[df_aa['target_predetto_adv'] == 'Vulnerabile']
df_pi_fail = df_pi[df_pi['target_predetto_pj'] == 'Vulnerabile']
df_ap_fail = df_ap[df_ap['target_predetto_ap'] == 'Vulnerabile']

N_CAMPIONI = 200 

risultati_totali = []

for model_name, config in models_config.items():
    print(f"\n{'='*70}\n ESTREZIONE NEURO-SEMANTICA (TUTTI I LAYER): {model_name}\n{'='*70}")
    
    df_mod = df_baseline[(df_baseline['modello'] == model_name) & (df_baseline['target_vero'] == 'Vulnerabile')]
    if len(df_mod) == 0:
        continue

    nome_file_safe = model_name.replace('/', '_')

    codici_aa = df_aa_fail[df_aa_fail['modello'] == model_name]['codice_originale'].tolist()[:N_CAMPIONI]
    codici_pi = df_pi_fail[df_pi_fail['modello'] == model_name]['codice_originale'].tolist()[:N_CAMPIONI]
    codici_ap = df_ap_fail[df_ap_fail['modello'] == model_name]['codice_originale'].tolist()[:N_CAMPIONI]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                low_cpu_mem_usage=True,
                dtype=torch.float16
            )

        shift_aa_list, shift_pj_list, shift_ap_list = [], [], []
        h_clean_aa_list, h_clean_pi_list, h_clean_ap_list = [], [], []

        def get_all_hidden_states(codice):
            prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
            messages = [
                {"role": "system", "content": "You are a cybersecurity expert. Your task is to find vulnerabilities in the source code."},
                {"role": "user", "content": prompt}
            ]
            testo = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            testo = testo + "FINAL_VERDICT:"
            inputs = tokenizer([testo], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            
            return torch.stack([layer_state[0, -1, :] for layer_state in out.hidden_states])

        print("Calcolo delle direzioni vettoriali su tutti i layer")
        # --- Advanced Adversarial ---
        for cod_orig in codici_aa:
            cod_hackerato = df_aa[(df_aa['modello'] == model_name) & (df_aa['codice_originale'] == cod_orig)]['codice_perturbato'].values[0]
            
            h_clean = get_all_hidden_states(cod_orig)
            h_adv = get_all_hidden_states(cod_hackerato)
            shift_aa_list.append(h_adv - h_clean)
            h_clean_aa_list.append(h_clean)

        # --- Prompt Injection ---
        for cod_orig in codici_pi:
            cod_hackerato = df_pi[(df_pi['modello'] == model_name) & (df_pi['codice_originale'] == cod_orig)]['codice_perturbato'].values[0]
            
            h_clean = get_all_hidden_states(cod_orig)
            h_pi = get_all_hidden_states(cod_hackerato)
            shift_pj_list.append(h_pi - h_clean)
            h_clean_pi_list.append(h_clean)

        # --- Adversarial Perturbation ---
        for cod_orig in codici_ap:
            cod_hackerato = df_ap[(df_ap['modello'] == model_name) & (df_ap['codice_originale'] == cod_orig)]['codice_perturbato'].values[0]
            
            h_clean = get_all_hidden_states(cod_orig)
            h_ap = get_all_hidden_states(cod_hackerato)
            shift_ap_list.append(h_ap - h_clean)
            h_clean_ap_list.append(h_clean)

        vettore_aa_all_layers = torch.mean(torch.stack(shift_aa_list), dim=0) if shift_aa_list else None
        vettore_pj_all_layers = torch.mean(torch.stack(shift_pj_list), dim=0) if shift_pj_list else None
        vettore_ap_all_layers = torch.mean(torch.stack(shift_ap_list), dim=0) if shift_ap_list else None
        
        v_base_aa_all = torch.mean(torch.stack(h_clean_aa_list), dim=0) if h_clean_aa_list else None
        v_base_pi_all = torch.mean(torch.stack(h_clean_pi_list), dim=0) if h_clean_pi_list else None
        v_base_ap_all = torch.mean(torch.stack(h_clean_ap_list), dim=0) if h_clean_ap_list else None

        # CONTRASTIVE LOGIT LENS 
        lm_head = model.get_output_embeddings() 
        final_layernorm = model.model.norm
        id_true = tokenizer.encode(" True", add_special_tokens=False)[-1]
        id_false = tokenizer.encode(" False", add_special_tokens=False)[-1]
        w_true = lm_head.weight[id_true]
        w_false = lm_head.weight[id_false]

        def calcola_delta(vettore_layer):
            if vettore_layer is None: return np.nan
            v_calc = vettore_layer.to(model.dtype)
            
            v_calc_scaled = v_calc * final_layernorm.weight
            
            logit_true = torch.dot(v_calc_scaled, w_true).item()
            logit_false = torch.dot(v_calc_scaled, w_false).item()
            return logit_true - logit_false

        num_layers_totali = vettore_aa_all_layers.shape[0] if vettore_aa_all_layers is not None else len(model.model.layers) + 1

        print(f"Eseguendo Logit Lens su {num_layers_totali} layer")
        
        for layer_idx in range(num_layers_totali):
            
            path_steering = f"attivazioni_totali/steering_vector_{nome_file_safe}_layer_{layer_idx}.npy"
            vettore_steering_layer = None
            if os.path.exists(path_steering):
                vettore_steering_layer = torch.tensor(np.load(path_steering), dtype=model.dtype, device=model.device)

            v_aa_layer = vettore_aa_all_layers[layer_idx] if vettore_aa_all_layers is not None else None
            v_pj_layer = vettore_pj_all_layers[layer_idx] if vettore_pj_all_layers is not None else None
            v_ap_layer = vettore_ap_all_layers[layer_idx] if vettore_ap_all_layers is not None else None

            # calcolo i 4 Delta
            delta_steer = calcola_delta(vettore_steering_layer)
            delta_aa = calcola_delta(v_aa_layer)
            delta_pj = calcola_delta(v_pj_layer)
            delta_ap = calcola_delta(v_ap_layer)

            base_aa = calcola_delta(v_base_aa_all[layer_idx] if v_base_aa_all is not None else None)
            base_pi = calcola_delta(v_base_pi_all[layer_idx] if v_base_pi_all is not None else None)
            base_ap = calcola_delta(v_base_ap_all[layer_idx] if v_base_ap_all is not None else None)
            risultati_totali.append({
                "MODELLO": model_name,
                "LAYER": layer_idx,
                "Delta_Steering": delta_steer,
                "Delta_AA": delta_aa,
                "Delta_PI": delta_pj,
                "Delta_AP": delta_ap,
                "Base_AA": base_aa,
                "Base_PI": base_pi,
                "Base_AP": base_ap
            })

        print("Salvataggio dati layer completato.")

        del model
        del tokenizer
        
        del shift_aa_list, shift_pj_list, shift_ap_list
        del vettore_aa_all_layers, vettore_pj_all_layers, vettore_ap_all_layers
        
    
        import gc
        gc.collect()
        gc.collect() 
        
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Errore su {model_name}: {e}")

percorso_csv_finale = "CSV tesi/Logit_Lens/Risultati_Contrastive_All_Layers_fallimenti.csv"
df_risultati = pd.DataFrame(risultati_totali)
df_risultati.to_csv(percorso_csv_finale, index=False)

print(f"\n{'='*70}\n ELABORAZIONE FINITA!")
print(f" Tutti i risultati sono stati salvati in 5 colonne in: {percorso_csv_finale}\n{'='*70}")