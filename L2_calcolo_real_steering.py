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

# --- 1. CONFIGURAZIONE DEGLI IPERPARAMETRI ---
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

# --- 2. DEFINIZIONE DELL'INTERVENTO A TEMPO DI INFERENZA (ITI) ---
def crea_hook_offensiva(vettore_tensore, moltiplicatore):
    vettore_norm = vettore_tensore / torch.norm(vettore_tensore)
    
    def steering_hook_offensiva(module, input, output):
        tensore_modificato = output[0].clone() if isinstance(output, tuple) else output.clone()
        vettore_locale = vettore_norm.to(tensore_modificato.device)
        
        if len(tensore_modificato.shape) == 3:
            tensore_modificato[:, -1, :] = tensore_modificato[:, -1, :] - (vettore_locale * moltiplicatore)
            
        return (tensore_modificato,) + output[1:] if isinstance(output, tuple) else tensore_modificato
    return steering_hook_offensiva

df_modello_TRUE = pd.read_csv("CSV tesi/Dataset/dataset_TRUE.csv")

# --- 3. VALUTAZIONE SPERIMENTALE MULTI-MODELLO ---
for model_name, config in models_config.items():
    print(f"\n{'='*50}\nElaborazione VERO COLLASSO L2 (Triangoli): {model_name}\n{'='*50}")
    
    df_corretti = df_modello_TRUE[df_modello_TRUE['modello'] == model_name]
    if len(df_corretti) == 0:
        continue

    nome_file_safe = model_name.replace('/', '_')
    layer_iniezione = layer_per_modello.get(model_name)[0]  
    
    moltiplicatore = moltiplicatori_per_modello.get(model_name)[0] 

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        dtype=torch.float16
    )

    num_layers = len(model.model.layers)
    
    percorso_vettore = f"attivazioni_totali/steering_vector_{nome_file_safe}_layer_{layer_iniezione}.npy"
    if not os.path.exists(percorso_vettore):
        print(f"File vettore non trovato per {model_name}, salto.")
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        continue
        
    vettore_numpy = np.load(percorso_vettore)
    vettore_tensore = torch.tensor(vettore_numpy, dtype=model.dtype, device=model.device)

    attivazioni_vulnerabili_steered = {i: [] for i in range(num_layers)}
    attivazioni_sicuri_vanilla = {i: [] for i in range(num_layers)}

    # --- 4. ESTRAZIONE DELLE RAPPRESENTAZIONI LATENTI ---
    for index, row in df_corretti.iterrows():
        codice = str(row['codice'])
        target = row['target_vero']

        prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
        messages = [{"role": "system", "content": "You are a cybersecurity expert."}, {"role": "user", "content": prompt}]
        testo_formattato = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([testo_formattato], return_tensors="pt").to(model.device)

        if target == 'Sicuro':
            with torch.no_grad():
                outputs_vanilla = model(**inputs, output_hidden_states=True)
            hidden_vanilla = outputs_vanilla.hidden_states[1:] 
            for layer_idx in range(num_layers):
                attivazioni_sicuri_vanilla[layer_idx].append(hidden_vanilla[layer_idx][0, -1, :].detach().cpu().numpy())
            del outputs_vanilla, hidden_vanilla

        elif target == 'Vulnerabile':
            hook_handle = model.model.layers[layer_iniezione].register_forward_hook(
                crea_hook_offensiva(vettore_tensore, moltiplicatore)
            )
            with torch.no_grad():
                outputs_steered = model(**inputs, output_hidden_states=True)
            hook_handle.remove() 
            
            hidden_steered = outputs_steered.hidden_states[1:] 
            for layer_idx in range(num_layers):
                attivazioni_vulnerabili_steered[layer_idx].append(hidden_steered[layer_idx][0, -1, :].detach().cpu().numpy())
            del outputs_steered, hidden_steered
            
        del inputs

    # --- 5. CALCOLO GEOMETRICO DELLO SHIFT LATENTE (TRIANGOLO COLLASSO) ---
    dist_Sv_Vv = [] # Baseline
    dist_Sv_Vs = [] # Distanza residua (Il collasso verso lo zero)
    dist_Vv_Vs = [] # Entità della spinta dello steering

    for layer_idx in range(num_layers):
        if len(attivazioni_vulnerabili_steered[layer_idx]) == 0 or len(attivazioni_sicuri_vanilla[layer_idx]) == 0:
            dist_Sv_Vv.append(0); dist_Sv_Vs.append(0); dist_Vv_Vs.append(0)
            continue
            
        S_v = np.mean(np.stack(attivazioni_sicuri_vanilla[layer_idx]), axis=0)
        V_s = np.mean(np.stack(attivazioni_vulnerabili_steered[layer_idx]), axis=0)
        
        # Recuperiamo la Baseline V_v dal vettore vanilla salvato in precedenza
        # Vettore Vanilla = V_v - S_v  -->  V_v = Vettore Vanilla + S_v
        percorso_vettore_vanilla = f"attivazioni_totali/steering_vector_{nome_file_safe}_layer_{layer_idx}.npy"
        if os.path.exists(percorso_vettore_vanilla):
            vec_vanilla = np.load(percorso_vettore_vanilla) 
            V_v = vec_vanilla + S_v 
            dist_Sv_Vv.append(np.linalg.norm(vec_vanilla)) # Che equivale a norm(V_v - S_v)
            dist_Vv_Vs.append(np.linalg.norm(V_s - V_v))
        else:
            dist_Sv_Vv.append(0)
            dist_Vv_Vs.append(0)

        dist_Sv_Vs.append(np.linalg.norm(V_s - S_v))

    # --- 6. VISUALIZZAZIONE DEI RISULTATI ---
    plt.figure(figsize=(10, 6))
    if any(dist_Sv_Vv):
        plt.plot(range(num_layers), dist_Sv_Vv, marker='o', linestyle='-', color='blue', label='Baseline (Vulnerabile vs Sicuro Vanilla)', alpha=0.5)

    plt.plot(range(num_layers), dist_Sv_Vs, marker='x', linestyle='-', color='red', label=f'Residuo Post-Steering (Iniezione a L{layer_iniezione})', linewidth=2)
    plt.axvline(x=layer_iniezione, color='black', linestyle='--', label='Punto di Iniezione Steering')

    plt.title(f'Vero Collasso della Vulnerabilità - {model_name}')
    plt.xlabel('Indice del Layer')
    plt.ylabel('Distanza dal Sicuro Vanilla (L2)')
    plt.legend()
    plt.grid(True)

    percorso_grafico = f"{nome_file_safe}_Vero_Collasso.png"
    plt.savefig(percorso_grafico)
    plt.close()
    print(f"Grafico salvato in: {percorso_grafico}")

    # ==========================================
    # 7. ESPORTAZIONE DATI GREZZI IN CSV
    # ==========================================
    print("Esportazione valori L2 grezzi in CSV...")
    os.makedirs("CSV tesi/Dati_Grafici_L2", exist_ok=True)
    
    df_export = pd.DataFrame({
        'Layer': range(num_layers),
        'Baseline_Sv_Vv': dist_Sv_Vv,
        'Sv_Vs (Residuo da Sicuro)': dist_Sv_Vs,
        'Vv_Vs (Shift dello Steering)': dist_Vv_Vs
    })
    
    csv_path = f"CSV tesi/Dati_Grafici_L2/{nome_file_safe}_Steering_L2.csv"
    df_export.to_csv(csv_path, index=False)
    print(f" -> Dati numerici salvati in: {csv_path}")

    try:
        del model, tokenizer
        del attivazioni_vulnerabili_steered, attivazioni_sicuri_vanilla
    except NameError:
        pass
    gc.collect()
    torch.cuda.empty_cache()