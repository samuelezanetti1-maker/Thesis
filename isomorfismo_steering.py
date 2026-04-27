import os
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import models_config

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/scratch_share/bislab/HF_HUB_CACHE/"

# 1. CARICAMENTO DATI
df_TP = pd.read_csv("CSV tesi/Dataset/dataset_TP.csv")
df_steering_res = pd.read_csv("CSV tesi/risultati_attacco_medie_steering.csv")

# Mapping
mapping_dict = df_TP['codice'].to_dict()
df_steering_res['codice_originale'] = df_steering_res['id_snippet'].map(mapping_dict)

layer_locus_per_modello = {
    "Qwen/Qwen2.5-7B-Instruct": 18,
    "Qwen/Qwen2.5-Coder-7B-Instruct": 18,
    "meta-llama/Llama-3.1-8B-Instruct": 15,
    "codellama/CodeLlama-7b-Instruct-hf": 13,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 19,
    "deepseek-ai/deepseek-coder-6.7b-instruct": 16
}

risultati_isomorfismo = []

def crea_hook_offensiva(vettore_tensore, moltiplicatore):
    vettore_norm = vettore_tensore / torch.norm(vettore_tensore)

    def steering_hook_offensiva(module, input, output):
        if isinstance(output, tuple):
            tensore_modificato = output[0].clone()
        else:
            tensore_modificato = output.clone()

        vettore_locale = vettore_norm.to(tensore_modificato.device)
        
        if len(tensore_modificato.shape) == 3:
            # SOTTRAZIONE: il cos_sim sarà -1.0
            tensore_modificato[:, -1, :] = tensore_modificato[:, -1, :] - (vettore_locale * moltiplicatore)
        elif len(tensore_modificato.shape) == 2:
            tensore_modificato = tensore_modificato - (vettore_locale * moltiplicatore)
            
        if isinstance(output, tuple):
            return (tensore_modificato,) + output[1:]
        else:
            return tensore_modificato
    return steering_hook_offensiva

for model_name, config in models_config.items():
    df_mod = df_steering_res[
        (df_steering_res['modello'] == model_name) & 
        (df_steering_res['predizione_post_steering'] == 'Sicuro')
    ]

    if len(df_mod) == 0:
        continue

    print(f"\nAnalisi Isomorfismo su: {model_name} ({len(df_mod)} successi)")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
        
        # --- FIX: Variabili spostate QUI, dove 'model' esiste ---
        totale_layer = len(model.model.layers)
        layer_penultimo = totale_layer - 2
        layer_locus = layer_locus_per_modello.get(model_name)
        moltiplicatore = df_mod.iloc[0]['moltiplicatore']
        
        # --- FIX: Carichiamo il vettore del LOCUS, non del 55% ---
        nome_mod_pulito = model_name.replace("/", "_")
        path_vettore = f"attivazioni_totali/steering_vector_{nome_mod_pulito}_layer_{layer_locus}.npy"
        
        vettore_steering_np = np.load(path_vettore)
        vettore_tensore = torch.tensor(vettore_steering_np, dtype=model.dtype, device=model.device)

        for index, row in df_mod.iterrows():
            codice_pulito = str(row['codice_originale']) 
            
            prompt = f"Analyze this code \n\nCode:\n{codice_pulito}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
            messages = [{"role": "system", "content": "You are a cybersecurity expert."}, {"role": "user", "content": prompt}]
            testo_formattato = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([testo_formattato], return_tensors="pt").to(model.device)
            
            # PASSAGGIO 1: Forward Pass Pulita
            with torch.no_grad():
                outputs_clean = model(**inputs, output_hidden_states=True)
            
            h_clean_locus = outputs_clean.hidden_states[layer_locus][0, -1, :]
            h_clean_penultimo = outputs_clean.hidden_states[layer_penultimo][0, -1, :]

            # PASSAGGIO 2: Forward Pass con Activation Steering al layer locus
            handle = model.model.layers[layer_locus].register_forward_hook(
                crea_hook_offensiva(vettore_tensore, moltiplicatore)
            )
            
            with torch.no_grad():
                outputs_steered = model(**inputs, output_hidden_states=True)
                
            handle.remove()
            
            h_steered_locus = outputs_steered.hidden_states[layer_locus][0, -1, :]
            h_steered_penultimo = outputs_steered.hidden_states[layer_penultimo][0, -1, :]

            # PASSAGGIO 3: Calcolo
            shift_locus = h_steered_locus - h_clean_locus
            cos_sim_locus = F.cosine_similarity(shift_locus, vettore_tensore, dim=0).item()
            
            shift_penultimo = h_steered_penultimo - h_clean_penultimo
            cos_sim_penultimo = F.cosine_similarity(shift_penultimo, vettore_tensore, dim=0).item()

            risultati_isomorfismo.append({
                "id_snippet": row['id_snippet'], # Meglio usare l'id originale invece di index
                "modello": model_name,
                "layer_locus": layer_locus,
                "cos_sim_stesso_layer": cos_sim_locus,        # Sarà vicino a -1.0
                "cos_sim_penultimo_layer": cos_sim_penultimo  
            })

        del model, tokenizer
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Errore su {model_name}: {e}")

pd.DataFrame(risultati_isomorfismo).to_csv("CSV tesi/Isomorfismo_Steering.csv", index=False)