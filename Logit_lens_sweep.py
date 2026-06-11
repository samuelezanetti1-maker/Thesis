import os
import pandas as pd
import torch
import gc
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import models_config

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/scratch_share/bislab/HF_HUB_CACHE/"

df_baseline = pd.read_csv("CSV tesi/Dataset/dataset_TRUE.csv")
os.makedirs("CSV tesi/Logit_Lens", exist_ok=True) 

df_aa = pd.read_csv("CSV tesi/Fixed/risultati_attacco_advanced.csv")
df_pi = pd.read_csv("CSV tesi/Fixed/risultati_attacco_PJ.csv")
df_ap = pd.read_csv("CSV tesi/Fixed/risultati_attacco_adversarial_perturbation.csv")

# Filtro SUCCESSI
df_aa_succ = df_aa[df_aa['target_predetto_adv'] == 'Sicuro']
df_pi_succ = df_pi[df_pi['target_predetto_pj'] == 'Sicuro']
df_ap_succ = df_ap[df_ap['target_predetto_ap'] == 'Sicuro']

risultati_totali = []

for model_name, config in models_config.items():
    print(f"\n{'='*70}\n ESTRAZIONE NEURO-SEMANTICA (TUTTI I LAYER): {model_name}\n{'='*70}")
    
    df_mod = df_baseline[(df_baseline['modello'] == model_name) & (df_baseline['target_vero'] == 'Vulnerabile')]
    if len(df_mod) == 0:
        continue

    nome_file_safe = model_name.replace('/', '_')

    codici_aa = df_aa_succ[df_aa_succ['modello'] == model_name]['codice_originale'].tolist()
    codici_pi = df_pi_succ[df_pi_succ['modello'] == model_name]['codice_originale'].tolist()
    codici_ap = df_ap_succ[df_ap_succ['modello'] == model_name]['codice_originale'].tolist()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                low_cpu_mem_usage=True,
                dtype=torch.float16
            )

        # Inizializziamo i pesi per la Logit Lens una volta sola
        lm_head = model.get_output_embeddings() 
        final_layernorm = model.model.norm
        id_true = tokenizer.encode(" True", add_special_tokens=False)[-1]
        id_false = tokenizer.encode(" False", add_special_tokens=False)[-1]
        w_true = lm_head.weight[id_true]
        w_false = lm_head.weight[id_false]

        # LA MAGIA È QUI: Calcoliamo i logit DIRETTAMENTE per ogni livello
        def estrai_logits_per_layer(codice):
            prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
            messages = [
                {"role": "system", "content": "You are a cybersecurity expert. Your task is to find vulnerabilities in the source code."},
                {"role": "user", "content": prompt}
            ]
            testo_base = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            if "DeepSeek-R1" in model_name:
                inputs_gen = tokenizer([testo_base], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(**inputs_gen, max_new_tokens=600, pad_token_id=tokenizer.eos_token_id, do_sample=False)
                token_generati = output_ids[0][inputs_gen.input_ids.shape[1]:]
                testo_generato = tokenizer.decode(token_generati, skip_special_tokens=False)
                pensiero_puro = testo_generato.split("</think>")[0] + "</think>\n" if "</think>" in testo_generato else testo_generato
                testo_finale = testo_base + pensiero_puro + "FINAL_VERDICT:"
            else:
                testo_finale = testo_base + "FINAL_VERDICT:"

            inputs = tokenizer([testo_finale], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            
            # Calcoliamo i logit subitissimo, tenendo solo un NUMERO in memoria
            layer_logits = []
            for layer_state in out.hidden_states:
                v_calc = layer_state[0, -1, :].to(model.dtype)
                v_calc_norm = final_layernorm(v_calc)  # Applichiamo la norma al tensore puro
                logit_true = torch.dot(v_calc_norm, w_true).item()
                logit_false = torch.dot(v_calc_norm, w_false).item()
                layer_logits.append(logit_true - logit_false) # Salviamo solo il Delta
                
            return layer_logits

        print("Calcolo dei Logit puntuali su tutti i layer...")
        
        # Salviamo solo liste di float (risparmio estremo di RAM!)
        logits_clean_aa, logits_adv_aa = [], []
        logits_clean_pi, logits_adv_pi = [], []
        logits_clean_ap, logits_adv_ap = [], []

        for cod_orig in codici_aa:
            cod_hackerato = df_aa_succ[(df_aa_succ['modello'] == model_name) & (df_aa_succ['codice_originale'] == cod_orig)]['codice_perturbato'].values[0]
            logits_clean_aa.append(estrai_logits_per_layer(cod_orig))
            logits_adv_aa.append(estrai_logits_per_layer(cod_hackerato))

        for cod_orig in codici_pi:
            cod_hackerato = df_pi_succ[(df_pi_succ['modello'] == model_name) & (df_pi_succ['codice_originale'] == cod_orig)]['codice_perturbato'].values[0]
            logits_clean_pi.append(estrai_logits_per_layer(cod_orig))
            logits_adv_pi.append(estrai_logits_per_layer(cod_hackerato))

        for cod_orig in codici_ap:
            cod_hackerato = df_ap_succ[(df_ap_succ['modello'] == model_name) & (df_ap_succ['codice_originale'] == cod_orig)]['codice_perturbato'].values[0]
            logits_clean_ap.append(estrai_logits_per_layer(cod_orig))
            logits_adv_ap.append(estrai_logits_per_layer(cod_hackerato))

        # Medie aritmetiche finali (Media matematica sicura sui numeri scalari)
        mean_base_aa = np.mean(logits_clean_aa, axis=0) if logits_clean_aa else []
        mean_adv_aa = np.mean(logits_adv_aa, axis=0) if logits_adv_aa else []
        
        mean_base_pi = np.mean(logits_clean_pi, axis=0) if logits_clean_pi else []
        mean_adv_pi = np.mean(logits_adv_pi, axis=0) if logits_adv_pi else []
        
        mean_base_ap = np.mean(logits_clean_ap, axis=0) if logits_clean_ap else []
        mean_adv_ap = np.mean(logits_adv_ap, axis=0) if logits_adv_ap else []

        num_layers_totali = len(model.model.layers) + 1
        print(f"-> Mappatura Logit Lens su {num_layers_totali} layer completata.")
        
        for layer_idx in range(num_layers_totali):
            
            # Estrazione sicura
            base_aa = mean_base_aa[layer_idx] if len(mean_base_aa) > 0 else np.nan
            abs_adv_aa = mean_adv_aa[layer_idx] if len(mean_adv_aa) > 0 else np.nan
            
            base_pi = mean_base_pi[layer_idx] if len(mean_base_pi) > 0 else np.nan
            abs_adv_pi = mean_adv_pi[layer_idx] if len(mean_adv_pi) > 0 else np.nan
            
            base_ap = mean_base_ap[layer_idx] if len(mean_base_ap) > 0 else np.nan
            abs_adv_ap = mean_adv_ap[layer_idx] if len(mean_adv_ap) > 0 else np.nan

            # Calcolo del puro shift (già operiamo su differenze di logit)
            delta_aa = abs_adv_aa - base_aa if not pd.isna(abs_adv_aa) else np.nan
            delta_pi = abs_adv_pi - base_pi if not pd.isna(abs_adv_pi) else np.nan
            delta_ap = abs_adv_ap - base_ap if not pd.isna(abs_adv_ap) else np.nan

            risultati_totali.append({
                "MODELLO": model_name,
                "LAYER": layer_idx,
                "Delta_AA": delta_aa,
                "Delta_PI": delta_pi,
                "Delta_AP": delta_ap,
                "Base_AA": base_aa,
                "Base_PI": base_pi,
                "Base_AP": base_ap
            })

        print("Salvataggio dati layer completato.")

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Errore su {model_name}: {e}")

percorso_csv_finale = "CSV tesi/Logit_Lens/Risultati_Contrastive_All_Layers.csv"
df_risultati = pd.DataFrame(risultati_totali)
df_risultati.to_csv(percorso_csv_finale, index=False)

print(f"\n{'='*70}\n ELABORAZIONE FINITA!")
print(f" Tutti i risultati sono stati salvati in 5 colonne in: {percorso_csv_finale}\n{'='*70}")