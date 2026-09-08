from config import models_config
import os
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 
os.environ["HF_HOME"] = 

df_attacchi = pd.read_csv("CSV tesi/Fixed/risultati_attacco_adversarial_perturbation.csv")
df_aa_succ = df_attacchi[df_attacchi['target_predetto_ap'] == 'Sicuro']

risultati_isomorfismo = []

for model_name, config in models_config.items():
    
    df_mod = df_aa_succ[df_aa_succ['modello'] == model_name]
    if len(df_mod) == 0:
        continue

    nome_modello_pulito = model_name.replace("/", "_")
    print("\n" + "="*70)
    print(f" SWEEP ISOMORFISMO (VETTORI MULTIPLI) SU: {model_name} ({len(df_mod)} attacchi)")
    print("="*70)

    try:
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
            print("Nessun vettore trovato. Salto modello.")
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

        accumulatore_sim = {i: [] for i in range(num_layers_totali)}

        for index, row in df_mod.iterrows():
            codice_pulito = str(row['codice_originale']) 
            codice_hackerato = str(row['codice_perturbato'])

            # Fotografa tutti i layer
            h_clean = get_all_hidden_states(codice_pulito)
            h_adv = get_all_hidden_states(codice_hackerato)

            # Shift di tutti i layer
            shift_all = h_adv - h_clean

            # Calcolo Cosine Similarity: Layer N contro Vettore Steering N
            for layer_idx in range(num_layers_totali):
                vettore_riferimento = dizionario_vettori[layer_idx]
                
                # Calcoliamo solo se il vettore per quel layer esiste
                if vettore_riferimento is not None:
                    shift_layer = shift_all[layer_idx]
                    cos_sim = F.cosine_similarity(shift_layer, vettore_riferimento, dim=0).item()
                    accumulatore_sim[layer_idx].append(cos_sim)
        
        # Calcolo delle medie
        for layer_idx in range(num_layers_totali):
            valori_layer = accumulatore_sim[layer_idx]
            if len(valori_layer) > 0:
                media_sim = np.mean(valori_layer)
                media_assoluta = np.mean([abs(x) for x in valori_layer])
                
                risultati_isomorfismo.append({
                    "modello": model_name,
                    "layer_analizzato": layer_idx,
                    "media_cosine_similarity": media_sim,
                    "media_valore_assoluto": media_assoluta
                })
        
        del model, tokenizer
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Errore su {model_name}: {e}")
        continue

os.makedirs("CSV tesi/Isomorfismo_Sweep", exist_ok=True)
df_finale = pd.DataFrame(risultati_isomorfismo)
df_finale.to_csv("CSV tesi/Isomorfismo_Sweep/sweep_isomorfismo_multi_vector_AP.csv", index=False)
print("\n Salvataggio completato in 'CSV tesi/Isomorfismo_Sweep/sweep_isomorfismo_multi_vector_AP.csv'")
