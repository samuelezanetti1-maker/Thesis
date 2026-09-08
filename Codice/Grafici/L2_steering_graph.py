import os
import pandas as pd
import torch
import gc
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import models_config 

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 
os.environ["HF_HOME"] = 

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
        
        # Assicura che la "siringa" (vettore) sia sulla stessa GPU del tensore
        vettore_locale = vettore_norm.to(tensore_modificato.device)
        
        # Seleziona l'ultimo token generato ([-1]) e sottrae la direzione moltiplicata
        if len(tensore_modificato.shape) == 3:
            tensore_modificato[:, -1, :] = tensore_modificato[:, -1, :] - (vettore_locale * moltiplicatore)
            
        # Restituisce il pensiero alterato al flusso della rete neurale
        return (tensore_modificato,) + output[1:] if isinstance(output, tuple) else tensore_modificato
        
    return steering_hook_offensiva

# 2 Carica il dataset contente i codici sorgente
df_modello_TRUE = pd.read_csv("CSV tesi/Dataset/dataset_TRUE.csv")

# --- 3. CICLO SUI MODELLI ---
for model_name, config in models_config.items():
    print(f"\n{'='*50}\nElaborazione Grafico L2: {model_name}\n{'='*50}")
    
    # Filtra solo i codici in cui il modello ha risposto bene nella baseline (True Positives/True Negatives)
    df_corretti = df_modello_TRUE[df_modello_TRUE['modello'] == model_name]
    if len(df_corretti) == 0:
        continue

    # Crea un nome sicuro per salvare i file evitando conflitti con gli slash 
    nome_file_safe = model_name.replace('/', '_')
    
    # Estrae il layer bersaglio e il moltiplicatore dai dizionari definiti sopra
    layer_iniezione = layer_per_modello.get(model_name)[0]  
    lista_moltiplicatori = moltiplicatori_per_modello.get(model_name)
    moltiplicatore = lista_moltiplicatori[0] 

    # Accende il Tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto", # Distribuisce automaticamente sui core della GPU
        low_cpu_mem_usage=True,
        dtype=torch.float16 # Usa pesi a 16 bit per non saturare la memoria
    )

    num_layers = len(model.model.layers)
    
    # Cerca il file .npy da usare come iniezio e
    percorso_vettore = f"attivazioni_totali/steering_vector_{nome_file_safe}_layer_{layer_iniezione}.npy"
    if not os.path.exists(percorso_vettore):
        print(f"File vettore non trovato per {model_name}, salto.")
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        continue
        
    # Carica l'array di numpy dal disco e lo converte in un Tensore di PyTorch pronto per la GPU
    vettore_numpy = np.load(percorso_vettore)
    vettore_tensore = torch.tensor(vettore_numpy, dtype=model.dtype, device=model.device)

    # --- 4. PIAZZAMENTO DELL'HOOK ---
    hook_handle = model.model.layers[layer_iniezione].register_forward_hook(
        crea_hook_offensiva(vettore_tensore, moltiplicatore)
    )

    attivazioni_vulnerabili_steered = {i: [] for i in range(num_layers)}
    attivazioni_sicuri_steered = {i: [] for i in range(num_layers)}

    # --- 5. ESTRAZIONE SOTTO ATTACCO ---
    for index, row in df_corretti.iterrows():
        codice = str(row['codice'])
        target = row['target_vero']

        prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
        messages = [
            {"role": "system", "content": "You are a cybersecurity expert. Your task is to find vulnerabilities in the source code."},
            {"role": "user", "content": prompt}
        ]

        # Applica il template corretto per quel modello 
        testo_formattato = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([testo_formattato], return_tensors="pt").to(model.device)

        with torch.no_grad(): 
            outputs = model(**inputs, output_hidden_states=True)

        # Salta il layer 0
        hidden_states = outputs.hidden_states[1:] 

        # Smista le attivazioni layer per layer 
        for layer_idx in range(num_layers):
            # Prende il vettore [0, -1, :], ovvero: [batch 0, ULTIMO token generato, tutti i neuroni]
            vettore_ultimo_token = hidden_states[layer_idx][0, -1, :].detach().cpu().numpy()
            
            if target == 'Vulnerabile':
                attivazioni_vulnerabili_steered[layer_idx].append(vettore_ultimo_token)
            elif target == 'Sicuro':
                attivazioni_sicuri_steered[layer_idx].append(vettore_ultimo_token)

    hook_handle.remove()

    # --- 6. CALCOLO DELLA NORMA L2 E GRAFICO ---
    magnitudo_layer_steered = []
    magnitudo_layer_vanilla = [] 

    for layer_idx in range(num_layers):
        if len(attivazioni_vulnerabili_steered[layer_idx]) == 0 or len(attivazioni_sicuri_steered[layer_idx]) == 0:
            magnitudo_layer_steered.append(0)
            magnitudo_layer_vanilla.append(0)
            continue
            
        # Calcola il punto medio concettuale (centroide) dei codici manipolati
        media_vuln_steered = np.mean(np.stack(attivazioni_vulnerabili_steered[layer_idx]), axis=0)
        media_sicuro_steered = np.mean(np.stack(attivazioni_sicuri_steered[layer_idx]), axis=0)
        
        # Calcola la distanza vettoriale tra i due nuovi concetti
        vettore_steering_steered = media_vuln_steered - media_sicuro_steered
        
        # Applica il Teorema di Pitagora 
        magnitudo_layer_steered.append(np.linalg.norm(vettore_steering_steered))

        # Va a ripescare dal disco la lunghezza "Vanilla" calcolata originariamente 
        percorso_vettore_vanilla = f"attivazioni_totali/steering_vector_{nome_file_safe}_layer_{layer_idx}.npy"
        if os.path.exists(percorso_vettore_vanilla):
            vec_vanilla = np.load(percorso_vettore_vanilla)
            magnitudo_layer_vanilla.append(np.linalg.norm(vec_vanilla))
        else:
            magnitudo_layer_vanilla.append(0)

    # Avvia la creazione dell'immagine 
    plt.figure(figsize=(10, 6))
    if any(magnitudo_layer_vanilla):
        plt.plot(range(num_layers), magnitudo_layer_vanilla, marker='o', linestyle='-', color='blue', label='Vanilla (Sano)', alpha=0.5)

    plt.plot(range(num_layers), magnitudo_layer_steered, marker='x', linestyle='-', color='red', label=f'Steered (Iniezione a L{layer_iniezione})', linewidth=2)

    plt.axvline(x=layer_iniezione, color='black', linestyle='--', label='Punto di Iniezione Steering')

    plt.title(f'Schiacciamento della Vulnerabilità sotto Steering - {model_name}')
    plt.xlabel('Indice del Layer')
    plt.ylabel('Magnitudo del Vettore Steering (Norma L2)')
    plt.legend()
    plt.grid(True)
    percorso_grafico = f"{nome_file_safe}_campana_comparativa.png"
    plt.savefig(percorso_grafico)
    plt.close()
    print(f"Grafico salvato in: {percorso_grafico}")

    try:
        del model, tokenizer, inputs, outputs
        del attivazioni_vulnerabili_steered, attivazioni_sicuri_steered
    except NameError:
        pass
    gc.collect() # Forza il Garbage Collector di Python
    torch.cuda.empty_cache() # Svuota la cache fisica della scheda video Nvidia
