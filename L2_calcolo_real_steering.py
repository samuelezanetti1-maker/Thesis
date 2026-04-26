import os
import pandas as pd
import torch
import gc
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import models_config 

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/scratch_share/bislab/HF_HUB_CACHE/"

# --- 1. CONFIGURAZIONE ---
moltiplicatori_per_modello = {
    "Qwen/Qwen2.5-7B-Instruct": [20],
    "Qwen/Qwen2.5-Coder-7B-Instruct": [30],
    "meta-llama/Llama-3.1-8B-Instruct": [3],
    "codellama/CodeLlama-7b-Instruct-hf": [15],
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": [10],
    "deepseek-ai/deepseek-coder-6.7b-instruct": [8]
}

layer_per_modello = {
    "Qwen/Qwen2.5-7B-Instruct": [18],
    "Qwen/Qwen2.5-Coder-7B-Instruct": [18],
    "meta-llama/Llama-3.1-8B-Instruct": [15],
    "codellama/CodeLlama-7b-Instruct-hf": [13],
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": [19],
    "deepseek-ai/deepseek-coder-6.7b-instruct": [16]
}

def crea_hook_offensiva(vettore_tensore, moltiplicatore):
    vettore_norm = vettore_tensore / torch.norm(vettore_tensore)
    def steering_hook_offensiva(module, input, output):
        tensore_modificato = output[0].clone() if isinstance(output, tuple) else output.clone()
        vettore_locale = vettore_norm.to(tensore_modificato.device)
        if len(tensore_modificato.shape) == 3:
            tensore_modificato[:, -1, :] = tensore_modificato[:, -1, :] - (vettore_locale * moltiplicatore)
        return (tensore_modificato,) + output[1:] if isinstance(output, tuple) else tensore_modificato
    return steering_hook_offensiva

def safe_norm(v1, v2):
    if v1 is not None and v2 is not None: return np.linalg.norm(v1 - v2)
    return np.nan

df_modello_TRUE = pd.read_csv("CSV tesi/Dataset/dataset_TRUE.csv")
df_steering_res = pd.read_csv("CSV tesi/risultati_attacco_steering.csv")

for model_name, config in models_config.items():
    print(f"\n{'='*50}\nElaborazione VERO COLLASSO L2: {model_name}\n{'='*50}")
    
    df_corretti = df_modello_TRUE[df_modello_TRUE['modello'] == model_name]
    if len(df_corretti) == 0: continue

    nome_file_safe = model_name.replace('/', '_')
    layer_iniezione = layer_per_modello.get(model_name)[0]  
    moltiplicatore = moltiplicatori_per_modello.get(model_name)[0] 

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage=True, dtype=torch.float16)
    num_layers = len(model.model.layers)
    
    percorso_vettore = f"attivazioni_totali/steering_vector_{nome_file_safe}_layer_{layer_iniezione}.npy"
    if not os.path.exists(percorso_vettore):
        print(f"File vettore non trovato per {model_name}, salto.")
        del model, tokenizer; gc.collect(); torch.cuda.empty_cache(); continue
        
    vettore_tensore = torch.tensor(np.load(percorso_vettore), dtype=model.dtype, device=model.device)

    # Dizionari per estrazioni live
    S_base_list = {i: [] for i in range(num_layers)}; V_base_list = {i: [] for i in range(num_layers)}
    S_Steering_list = {i: [] for i in range(num_layers)}; V_Steering_list = {i: [] for i in range(num_layers)}

    # Filttiamo i risultati dello steering per questo modello e questo moltiplicatore
    risultati_mod = df_steering_res[(df_steering_res['modello'] == model_name) & (df_steering_res['moltiplicatore'] == moltiplicatore)]

    # --- 4. ESTRAZIONE LIVE ---
    for index, row in df_corretti.iterrows():
        codice = str(row['codice'])
        target = row['target_vero']
        prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
        messages = [{"role": "system", "content": "You are a cybersecurity expert."}, {"role": "user", "content": prompt}]
        testo_formattato = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([testo_formattato], return_tensors="pt").to(model.device)

        if target == 'Sicuro':
            with torch.no_grad(): outputs = model(**inputs, output_hidden_states=True)
            for layer_idx in range(num_layers): S_base_list[layer_idx].append(outputs.hidden_states[1:][layer_idx][0, -1, :].detach().cpu().numpy())
            del outputs

        elif target == 'Vulnerabile':
            with torch.no_grad(): outputs_vanilla = model(**inputs, output_hidden_states=True)
            for layer_idx in range(num_layers): V_base_list[layer_idx].append(outputs_vanilla.hidden_states[1:][layer_idx][0, -1, :].detach().cpu().numpy())
            del outputs_vanilla

            # Controlla l'esito dello steering per questo snippet
            riga_res = risultati_mod[risultati_mod['id_snippet'] == index]
            if len(riga_res) > 0:
                esito = riga_res.iloc[0]['predizione_post_steering']
                
                # Applichiamo lo steering e registriamo dove finisce
                hook_handle = model.model.layers[layer_iniezione].register_forward_hook(crea_hook_offensiva(vettore_tensore, moltiplicatore))
                with torch.no_grad(): outputs_steered = model(**inputs, output_hidden_states=True)
                hook_handle.remove() 
                
                for layer_idx in range(num_layers):
                    vec_steered = outputs_steered.hidden_states[1:][layer_idx][0, -1, :].detach().cpu().numpy()
                    if esito == 'Sicuro':
                        S_Steering_list[layer_idx].append(vec_steered)
                    elif esito == 'Vulnerabile':
                        V_Steering_list[layer_idx].append(vec_steered)
                del outputs_steered
        del inputs

    # --- 5. CALCOLO GEOMETRICO ---
    df_export = pd.DataFrame({'Layer': range(num_layers)})

    for layer_idx in range(num_layers):
        S_base = np.mean(np.stack(S_base_list[layer_idx]), axis=0) if S_base_list[layer_idx] else None
        V_base = np.mean(np.stack(V_base_list[layer_idx]), axis=0) if V_base_list[layer_idx] else None
        S_Steering = np.mean(np.stack(S_Steering_list[layer_idx]), axis=0) if S_Steering_list[layer_idx] else None
        V_Steering = np.mean(np.stack(V_Steering_list[layer_idx]), axis=0) if V_Steering_list[layer_idx] else None
        
        df_export.at[layer_idx, 'Baseline_Vanilla (S_base - V_base)'] = safe_norm(S_base, V_base)
        df_export.at[layer_idx, 'Steering_Overshooting (S_Steering - S_base)'] = safe_norm(S_Steering, S_base)
        df_export.at[layer_idx, 'Steering_Forza (S_Steering - V_base)'] = safe_norm(S_Steering, V_base)
        df_export.at[layer_idx, 'Steering_Confine (S_Steering - V_Steering)'] = safe_norm(S_Steering, V_Steering)

    os.makedirs("CSV tesi/Dati_Grafici_L2", exist_ok=True)
    csv_path = f"CSV tesi/Dati_Grafici_L2/{nome_file_safe}_Steering_L2.csv"
    df_export.to_csv(csv_path, index=False)
    print(f" -> Dati salvati in: {csv_path}")

    try: del model, tokenizer
    except NameError: pass
    gc.collect()
    torch.cuda.empty_cache()