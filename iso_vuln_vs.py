from config import models_config
import os
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/scratch_share/bislab/HF_HUB_CACHE/"

# 1. Configurazione dei tre tipi di attacco
attacks_config = [
    {
        "id": "AA",
        "csv_in": "CSV tesi/Fixed/risultati_attacco_advanced.csv",
        "target_col": "target_predetto_adv",
        "csv_out": "CSV tesi/Isomorfismo_Sweep/sweep_isomorfismo_multi_vector_AA.csv"
    },
    {
        "id": "AP",
        "csv_in": "CSV tesi/Fixed/risultati_attacco_adversarial_perturbation.csv",
        "target_col": "target_predetto_ap",
        "csv_out": "CSV tesi/Isomorfismo_Sweep/sweep_isomorfismo_multi_vector_AP.csv"
    },
    {
        "id": "PJ",
        "csv_in": "CSV tesi/Fixed/risultati_attacco_PJ.csv",
        "target_col": "target_predetto_pj",
        "csv_out": "CSV tesi/Isomorfismo_Sweep/sweep_isomorfismo_multi_vector_PJ.csv"
    }
]

# 2. Pre-caricamento e filtraggio dei DataFrame 
dfs_attacchi = {}
for atk in attacks_config:
    if os.path.exists(atk["csv_in"]):
        df = pd.read_csv(atk["csv_in"])
        dfs_attacchi[atk["id"]] = df[df[atk["target_col"]] == 'Vulnerabile']
    else:
        print(f"[!] Attenzione: File {atk['csv_in']} non trovato.")
        dfs_attacchi[atk["id"]] = pd.DataFrame() # DataFrame vuoto se il file manca

# Dizionario per accumulare i risultati divisi per attacco
risultati_isomorfismo = {atk["id"]: [] for atk in attacks_config}

# 3. Loop principale sui modelli
for model_name, config in models_config.items():
    
    # Verifico quanti attacchi in totale (AA + AP + PJ) ci sono per questo specifico modello
    da_analizzare = {atk["id"]: dfs_attacchi[atk["id"]][dfs_attacchi[atk["id"]]['modello'] == model_name] for atk in attacks_config}
    tot_attacchi = sum(len(df) for df in da_analizzare.values())
    
    if tot_attacchi == 0:
        continue # Nessun attacco riuscito per questo modello in nessun dataset

    nome_modello_pulito = model_name.replace("/", "_")
    print("\n" + "="*70)
    print(f" SWEEP ISOMORFISMO SU: {model_name} (Totale attacchi combinati: {tot_attacchi})")
    print("="*70)

    try:
        # Carico Modello e Tokenizer UNA SOLA VOLTA
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=config.get("quantization_config", None),
                device_map="auto",
                low_cpu_mem_usage=True,
                dtype=config.get("dtype", torch.float16)
            )
        
        num_layers_totali = len(model.model.layers) + 1

        print("Pre-caricamento vettori di steering in VRAM...")
        dizionario_vettori = {}
        vettori_trovati = 0
        
        for layer_idx in range(num_layers_totali):
            path_steering = f"attivazioni_totali/steering_vector_{nome_modello_pulito}_layer_{layer_idx}.npy"
            if os.path.exists(path_steering):
                vettore_np = np.load(path_steering)
                dizionario_vettori[layer_idx] = torch.tensor(vettore_np, dtype=model.dtype, device=model.device)
                vettori_trovati += 1
            else:
                dizionario_vettori[layer_idx] = None
                
        print(f"Trovati {vettori_trovati}/{num_layers_totali} vettori per questo modello.")
        
        if vettori_trovati == 0:
            print(" [!] Nessun vettore trovato. Salto modello.")
            del model, tokenizer
            torch.cuda.empty_cache()
            continue

        def get_all_hidden_states(codice):
            prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
            messages = [
                {"role": "system", "content": "You are a cybersecurity expert. Your task is to find vulnerabilities in the source code."},
                {"role": "user", "content": prompt}
            ]
            testo = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([testo], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            return torch.stack([layer_state[0, -1, :] for layer_state in out.hidden_states])

        # 4. Itero sui 3 tipi di attacco 
        for atk in attacks_config:
            atk_id = atk["id"]
            df_mod = da_analizzare[atk_id]
            
            if len(df_mod) == 0:
                continue
                
            print(f" -> Elaborazione attacco {atk_id} ({len(df_mod)} samples)...")
            accumulatore_sim = {i: [] for i in range(num_layers_totali)}

            for index, row in df_mod.iterrows():
                codice_pulito = str(row['codice_originale']) 
                codice_hackerato = str(row['codice_perturbato'])

                # Fotografa tutti i layer
                h_clean = get_all_hidden_states(codice_pulito)
                h_adv = get_all_hidden_states(codice_hackerato)

                # Shift di tutti i layer
                shift_all = h_adv - h_clean

                # Calcolo Cosine Similarity
                for layer_idx in range(num_layers_totali):
                    vettore_riferimento = dizionario_vettori[layer_idx]
                    if vettore_riferimento is not None:
                        shift_layer = shift_all[layer_idx]
                        cos_sim = F.cosine_similarity(shift_layer, vettore_riferimento, dim=0).item()
                        accumulatore_sim[layer_idx].append(cos_sim)
            
            # Calcolo delle medie per questo specifico attacco
            for layer_idx in range(num_layers_totali):
                valori_layer = accumulatore_sim[layer_idx]
                if len(valori_layer) > 0:
                    media_sim = np.mean(valori_layer)
                    media_assoluta = np.mean([abs(x) for x in valori_layer])
                    
                    risultati_isomorfismo[atk_id].append({
                        "modello": model_name,
                        "layer_analizzato": layer_idx,
                        "media_cosine_similarity": media_sim,
                        "media_valore_assoluto": media_assoluta
                    })
        
        # 5. Pulizia VRAM prima di passare al prossimo modello
        del model, tokenizer
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Errore su {model_name}: {e}")
        continue

# 6. Salvataggio di tutti e tre i CSV finali
os.makedirs("CSV tesi/iso_vuln_vs", exist_ok=True)
for atk in attacks_config:
    if len(risultati_isomorfismo[atk["id"]]) > 0:
        df_finale = pd.DataFrame(risultati_isomorfismo[atk["id"]])
        df_finale.to_csv(atk["csv_out"], index=False)
        print(f"\n Salvataggio completato in '{atk['csv_out']}'")
