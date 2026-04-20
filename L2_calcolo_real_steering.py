import os
import pandas as pd
import torch
import gc
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import models_config 

# Ottimizzazione dell'allocazione di memoria CUDA per prevenire la frammentazione 
# durante l'elaborazione iterativa di Large Language Models (LLM).
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/scratch_share/bislab/HF_HUB_CACHE/"

# --- 1. CONFIGURAZIONE DEGLI IPERPARAMETRI ---
# Definizione empirica dei moltiplicatori di magnitudo (forza di iniezione) 
# calcolati tramite precedente Ablation Study per massimizzare l'Attack Success Rate (ASR).
moltiplicatori_per_modello = {
    "Qwen/Qwen2.5-7B-Instruct": [20],
    "Qwen/Qwen2.5-Coder-7B-Instruct": [30],
    "meta-llama/Llama-3.1-8B-Instruct": [3],
    "codellama/CodeLlama-7b-Instruct-hf": [15],
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": [10],
    "deepseek-ai/deepseek-coder-6.7b-instruct": [8]
}

# Definizione dei layer bersaglio (Punti di Iniezione) corrispondenti al picco 
# di suscettibilità causale identificato nella fase di Layer Sweep.
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
    """
    Genera una funzione di forward hook per l'Activation Steering.
    Normalizza il vettore di direzione concettuale e applica una perturbazione 
    vettoriale calcolata sugli hidden states durante il forward pass.
    """
    # Normalizzazione L2 del vettore per garantire un'applicazione scalare controllata dal moltiplicatore
    vettore_norm = vettore_tensore / torch.norm(vettore_tensore)
    
    def steering_hook_offensiva(module, input, output):
        # Clonazione del tensore in output per preservare l'integrità del grafo computazionale di PyTorch
        tensore_modificato = output[0].clone() if isinstance(output, tuple) else output.clone()
        vettore_locale = vettore_norm.to(tensore_modificato.device)
        
        # Alterazione dello spazio latente: sottrazione della direzione concettuale (Vulnerabilità)
        # unicamente sull'ultimo token della sequenza elaborata, influenzando la generazione successiva.
        if len(tensore_modificato.shape) == 3:
            tensore_modificato[:, -1, :] = tensore_modificato[:, -1, :] - (vettore_locale * moltiplicatore)
            
        return (tensore_modificato,) + output[1:] if isinstance(output, tuple) else tensore_modificato
    return steering_hook_offensiva

# Caricamento del Ground Truth dataset (istanze validate come True Positives/True Negatives)
df_modello_TRUE = pd.read_csv("CSV tesi/Dataset/dataset_TRUE.csv")

# --- 3. VALUTAZIONE SPERIMENTALE MULTI-MODELLO ---
for model_name, config in models_config.items():
    print(f"\n{'='*50}\nElaborazione Grafico VERO COLLASSO L2: {model_name}\n{'='*50}")
    
    df_corretti = df_modello_TRUE[df_modello_TRUE['modello'] == model_name]
    if len(df_corretti) == 0:
        continue

    nome_file_safe = model_name.replace('/', '_')
    layer_iniezione = layer_per_modello.get(model_name)[0]  
    
    lista_moltiplicatori = moltiplicatori_per_modello.get(model_name)
    moltiplicatore = lista_moltiplicatori[0] 

    # Inizializzazione del LLM e del tokenizer con precisione a 16-bit per ottimizzazione VRAM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        dtype=torch.float16
    )

    num_layers = len(model.model.layers)
    
    # Caricamento della direzione latente isolata pre-calcolata (Vettore di Steering)
    percorso_vettore = f"attivazioni_totali/steering_vector_{nome_file_safe}_layer_{layer_iniezione}.npy"
    if not os.path.exists(percorso_vettore):
        print(f"File vettore non trovato per {model_name}, salto.")
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        continue
        
    vettore_numpy = np.load(percorso_vettore)
    vettore_tensore = torch.tensor(vettore_numpy, dtype=model.dtype, device=model.device)

    # Strutture dati per la raccolta delle rappresentazioni latenti post-intervento
    attivazioni_vulnerabili_steered = {i: [] for i in range(num_layers)}
    attivazioni_sicuri_vanilla = {i: [] for i in range(num_layers)}

    # --- 4. ESTRAZIONE DELLE RAPPRESENTAZIONI LATENTI ---
    for index, row in df_corretti.iterrows():
        codice = str(row['codice'])
        target = row['target_vero']

        # Costruzione del prompt deterministico per costringere il modello a una classificazione binaria
        prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
        messages = [
            {"role": "system", "content": "You are a cybersecurity expert. Your task is to find vulnerabilities in the source code."},
            {"role": "user", "content": prompt}
        ]

        testo_formattato = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([testo_formattato], return_tensors="pt").to(model.device)

        # Logica di campionamento selettivo: estrazione della Baseline vs. estrazione sotto attacco
        if target == 'Sicuro':
            # Estrazione delle attivazioni di controllo (Vanilla Forward Pass)
            with torch.no_grad():
                outputs_vanilla = model(**inputs, output_hidden_states=True)
            hidden_vanilla = outputs_vanilla.hidden_states[1:] 
            for layer_idx in range(num_layers):
                vec = hidden_vanilla[layer_idx][0, -1, :].detach().cpu().numpy()
                attivazioni_sicuri_vanilla[layer_idx].append(vec)
            del outputs_vanilla, hidden_vanilla

        elif target == 'Vulnerabile':
            # Applicazione dell'Inference-Time Intervention (Steered Forward Pass)
            hook_handle = model.model.layers[layer_iniezione].register_forward_hook(
                crea_hook_offensiva(vettore_tensore, moltiplicatore)
            )
            with torch.no_grad():
                outputs_steered = model(**inputs, output_hidden_states=True)
            hook_handle.remove() # Rimozione immediata dell'hook per ripristinare lo stato intonso della rete
            
            hidden_steered = outputs_steered.hidden_states[1:] 
            for layer_idx in range(num_layers):
                vec = hidden_steered[layer_idx][0, -1, :].detach().cpu().numpy()
                attivazioni_vulnerabili_steered[layer_idx].append(vec)
            del outputs_steered, hidden_steered
            
        del inputs

    # --- 5. CALCOLO GEOMETRICO DELLO SHIFT LATENTE (TRUE COLLAPSE) ---
    magnitudo_layer_steered = []
    magnitudo_layer_vanilla = [] 

    for layer_idx in range(num_layers):
        if len(attivazioni_vulnerabili_steered[layer_idx]) == 0 or len(attivazioni_sicuri_vanilla[layer_idx]) == 0:
            magnitudo_layer_steered.append(0)
            magnitudo_layer_vanilla.append(0)
            continue
            
        # Calcolo dei centroidi nello spazio N-dimensionale (es. R^4096)
        media_vuln_steered = np.mean(np.stack(attivazioni_vulnerabili_steered[layer_idx]), axis=0)
        media_sicuro_vanilla = np.mean(np.stack(attivazioni_sicuri_vanilla[layer_idx]), axis=0)
        
        # Calcolo della deviazione residua tra le istanze vulnerabili manipolate e la baseline sicura
        vettore_vero_collasso = media_vuln_steered - media_sicuro_vanilla
        
        # Misurazione della Norma Euclidea (Distanza L2) per quantificare l'allineamento dei concetti
        magnitudo_layer_steered.append(np.linalg.norm(vettore_vero_collasso))

        # Recupero della magnitudo Vanilla pre-calcolata per l'analisi comparativa
        percorso_vettore_vanilla = f"attivazioni_totali/steering_vector_{nome_file_safe}_layer_{layer_idx}.npy"
        if os.path.exists(percorso_vettore_vanilla):
            vec_vanilla = np.load(percorso_vettore_vanilla)
            magnitudo_layer_vanilla.append(np.linalg.norm(vec_vanilla))
        else:
            magnitudo_layer_vanilla.append(0)

    # --- 6. VISUALIZZAZIONE DEI RISULTATI ---
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

    # ==========================================
    # 7. ESPORTAZIONE DATI GREZZI IN CSV
    # ==========================================
    print("Esportazione valori L2 grezzi in CSV...")
    os.makedirs("CSV tesi/Dati_Grafici_L2", exist_ok=True)
    
    df_export = pd.DataFrame({
        'Layer': range(num_layers),
        'Baseline_Vanilla': magnitudo_layer_vanilla,
        'Steered_Vero_Collasso': magnitudo_layer_steered
    })
    
    csv_path = f"CSV tesi/Dati_Grafici_L2/{nome_file_safe}_Steering_L2.csv"
    df_export.to_csv(csv_path, index=False)
    print(f" -> Dati numerici salvati in: {csv_path}")

    # Operazioni di deallocazione della memoria GPU per consentire l'esecuzione del modello successivo
    try:
        del model, tokenizer
        del attivazioni_vulnerabili_steered, attivazioni_sicuri_vanilla
    except NameError:
        pass
    gc.collect()
    torch.cuda.empty_cache()