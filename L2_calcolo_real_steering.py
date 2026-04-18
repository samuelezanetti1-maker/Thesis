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

df_modello_TRUE = pd.read_csv("CSV tesi/Dataset/dataset_TRUE.csv")

for model_name, config in models_config.items():
    print(f"\n{'='*50}\nElaborazione Grafico VERO COLLASSO L2: {model_name}\n{'='*50}")
    
    df_corretti = df_modello_TRUE[df_modello_TRUE['modello'] == model_name]
    if len(df_corretti) == 0:
        continue

    nome_file_safe = model_name.replace('/', '_')
    layer_iniezione = layer_per_modello.get(model_name)[0]  
    
    lista_moltiplicatori = moltiplicatori_per_modello.get(model_name)
    moltiplicatore = lista_moltiplicatori[0] 

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

    # ATTENZIONE: HOOK SPOSTATO DENTRO IL CICLO. Qui prepariamo solo gli scatoloni.
    # Nota il cambio di nome: raccogliamo i Sicuri VANILLA!
    attivazioni_vulnerabili_steered = {i: [] for i in range(num_layers)}
    attivazioni_sicuri_vanilla = {i: [] for i in range(num_layers)}

    for index, row in df_corretti.iterrows():
        codice = str(row['codice'])
        target = row['target_vero']

        prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
        messages = [
            {"role": "system", "content": "You are a cybersecurity expert. Your task is to find vulnerabilities in the source code."},
            {"role": "user", "content": prompt}
        ]

        testo_formattato = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([testo_formattato], return_tensors="pt").to(model.device)

        # ATTENZIONE: MODIFICA QUI - Logica selettiva per Vanilla vs Steered
        if target == 'Sicuro':
            # Risonanza VANILLA (Nessun hook)
            with torch.no_grad():
                outputs_vanilla = model(**inputs, output_hidden_states=True)
            hidden_vanilla = outputs_vanilla.hidden_states[1:] 
            for layer_idx in range(num_layers):
                vec = hidden_vanilla[layer_idx][0, -1, :].detach().cpu().numpy()
                attivazioni_sicuri_vanilla[layer_idx].append(vec)
            del outputs_vanilla, hidden_vanilla

        elif target == 'Vulnerabile':
            # Risonanza STEERED (Piazzo l'hook, faccio la passata, e lo tolgo subito)
            hook_handle = model.model.layers[layer_iniezione].register_forward_hook(
                crea_hook_offensiva(vettore_tensore, moltiplicatore)
            )
            with torch.no_grad():
                outputs_steered = model(**inputs, output_hidden_states=True)
            hook_handle.remove() # Togliamo l'hook immediatamente
            
            hidden_steered = outputs_steered.hidden_states[1:] 
            for layer_idx in range(num_layers):
                vec = hidden_steered[layer_idx][0, -1, :].detach().cpu().numpy()
                attivazioni_vulnerabili_steered[layer_idx].append(vec)
            del outputs_steered, hidden_steered
            
        del inputs

    # --- CALCOLO E GRAFICO ---
    magnitudo_layer_steered = []
    magnitudo_layer_vanilla = [] 

    for layer_idx in range(num_layers):
        if len(attivazioni_vulnerabili_steered[layer_idx]) == 0 or len(attivazioni_sicuri_vanilla[layer_idx]) == 0:
            magnitudo_layer_steered.append(0)
            magnitudo_layer_vanilla.append(0)
            continue
            
        # ATTENZIONE: MODIFICA QUI - La nuova matematica del vero schiacciamento
        media_vuln_steered = np.mean(np.stack(attivazioni_vulnerabili_steered[layer_idx]), axis=0)
        media_sicuro_vanilla = np.mean(np.stack(attivazioni_sicuri_vanilla[layer_idx]), axis=0)
        
        vettore_vero_collasso = media_vuln_steered - media_sicuro_vanilla
        magnitudo_layer_steered.append(np.linalg.norm(vettore_vero_collasso))

        # Ripeschiamo i dati vanilla vecchi per fare la linea blu di confronto
        percorso_vettore_vanilla = f"attivazioni_totali/steering_vector_{nome_file_safe}_layer_{layer_idx}.npy"
        if os.path.exists(percorso_vettore_vanilla):
            vec_vanilla = np.load(percorso_vettore_vanilla)
            magnitudo_layer_vanilla.append(np.linalg.norm(vec_vanilla))
        else:
            magnitudo_layer_vanilla.append(0)

    plt.figure(figsize=(10, 6))

    if any(magnitudo_layer_vanilla):
        plt.plot(range(num_layers), magnitudo_layer_vanilla, marker='o', linestyle='-', color='blue', label='Vanilla (Sano)', alpha=0.5)

    plt.plot(range(num_layers), magnitudo_layer_steered, marker='x', linestyle='-', color='red', label=f'Distanza Post-Steering (Iniezione a L{layer_iniezione})', linewidth=2)

    plt.axvline(x=layer_iniezione, color='black', linestyle='--', label='Punto di Iniezione Steering')

    plt.title(f'Vero Collasso della Vulnerabilità (Distanza da Vanilla Sicuro) - {model_name}')
    plt.xlabel('Indice del Layer')
    plt.ylabel('Distanza tra Vuln(Steered) e Sicuro(Vanilla) - L2')
    plt.legend()
    plt.grid(True)

    percorso_grafico = f"{nome_file_safe}_Vero_Collasso.png"
    plt.savefig(percorso_grafico)
    plt.close()
    print(f"Grafico salvato in: {percorso_grafico}")

    # Pulizia
    try:
        del model, tokenizer
        del attivazioni_vulnerabili_steered, attivazioni_sicuri_vanilla
    except NameError:
        pass
    gc.collect()
    torch.cuda.empty_cache()