import os
import pandas as pd
import torch
import gc
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Importa le configurazioni hardware/modelli dal tuo file esterno
from config import models_config 

# Ottimizzazione della memoria della GPU per evitare frammentazione durante i cicli lunghi
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Imposta la cartella in cui HuggingFace salva/cerca i modelli (evita di riscaricarli)
os.environ["HF_HOME"] = "/scratch_share/bislab/HF_HUB_CACHE/"

# --- 1. CONFIGURAZIONE ---
# Dizionario con la "dose di veleno" (forza dello steering) calibrata per ogni modello
moltiplicatori_per_modello = {
    "Qwen/Qwen2.5-7B-Instruct": [20],
    "Qwen/Qwen2.5-Coder-7B-Instruct": [30],
    "meta-llama/Llama-3.1-8B-Instruct": [3],
    "codellama/CodeLlama-7b-Instruct-hf": [15],
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": [10],
    "deepseek-ai/deepseek-coder-6.7b-instruct": [8]
}

# Dizionario con il "Punto di Iniezione" (il layer più vulnerabile trovato tramite Sweep)
layer_per_modello = {
    "Qwen/Qwen2.5-7B-Instruct": [18],
    "Qwen/Qwen2.5-Coder-7B-Instruct": [18],
    "meta-llama/Llama-3.1-8B-Instruct": [15],
    "codellama/CodeLlama-7b-Instruct-hf": [13],
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": [19],
    "deepseek-ai/deepseek-coder-6.7b-instruct": [16]
}

# --- 2. LA FUNZIONE CHIRURGICA (L'HOOK) ---
def crea_hook_offensiva(vettore_tensore, moltiplicatore):
    # Prima di tutto, normalizza la freccia (la fa diventare di lunghezza 1). 
    # Questo serve per applicare il moltiplicatore in modo matematicamente puro e controllato.
    vettore_norm = vettore_tensore / torch.norm(vettore_tensore)
    
    # Questa è la funzione vera e propria che PyTorch eseguirà al posto di blocco
    def steering_hook_offensiva(module, input, output):
        # Clona il tensore in uscita dal layer per non corrompere i gradienti di PyTorch
        tensore_modificato = output[0].clone() if isinstance(output, tuple) else output.clone()
        
        # Assicura che la "siringa" (vettore) sia sulla stessa GPU del "paziente" (tensore)
        vettore_locale = vettore_norm.to(tensore_modificato.device)
        
        # LA MANIPOLAZIONE: Seleziona l'ultimo token generato ([-1]) e SOTTRAE la direzione moltiplicata
        # Questa è la riga che fisicamente strattona l'idea verso il concetto di "Sicuro"
        if len(tensore_modificato.shape) == 3:
            tensore_modificato[:, -1, :] = tensore_modificato[:, -1, :] - (vettore_locale * moltiplicatore)
            
        # Restituisce il pensiero alterato al flusso della rete neurale
        return (tensore_modificato,) + output[1:] if isinstance(output, tuple) else tensore_modificato
        
    return steering_hook_offensiva

# Carica il dataset contente i codici sorgente
df_modello_TRUE = pd.read_csv("CSV tesi/Dataset/dataset_TRUE.csv")

# --- 3. CICLO SUI MODELLI ---
for model_name, config in models_config.items():
    print(f"\n{'='*50}\nElaborazione Grafico L2: {model_name}\n{'='*50}")
    
    # Filtra solo i codici in cui il modello ha risposto bene nella baseline (True Positives/True Negatives)
    df_corretti = df_modello_TRUE[df_modello_TRUE['modello'] == model_name]
    if len(df_corretti) == 0:
        continue

    # Crea un nome sicuro per salvare i file evitando conflitti con gli slash (es. meta-llama_Llama...)
    nome_file_safe = model_name.replace('/', '_')
    
    # Estrae il layer bersaglio e il moltiplicatore dai dizionari definiti sopra
    layer_iniezione = layer_per_modello.get(model_name)[0]  
    lista_moltiplicatori = moltiplicatori_per_modello.get(model_name)
    moltiplicatore = lista_moltiplicatori[0] 

    # Accende il Tokenizer (che trasforma le parole in numeri) e il Modello (il cervello)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto", # Distribuisce automaticamente sui core della GPU
        low_cpu_mem_usage=True,
        dtype=torch.float16 # Usa pesi a 16 bit per non saturare la memoria
    )

    num_layers = len(model.model.layers)
    
    # Cerca il file .npy (il Vettore Vanilla estratto giorni fa) da usare come veleno
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
    # Attacca fisicamente la funzione chirurgica al layer specificato. Salva il "telecomando" (hook_handle)
    hook_handle = model.model.layers[layer_iniezione].register_forward_hook(
        crea_hook_offensiva(vettore_tensore, moltiplicatore)
    )

    # Prepara gli "scatoloni" vuoti per raccogliere i pensieri alterati del modello
    attivazioni_vulnerabili_steered = {i: [] for i in range(num_layers)}
    attivazioni_sicuri_steered = {i: [] for i in range(num_layers)}

    # --- 5. ESTRAZIONE SOTTO ATTACCO (Risonanza Magnetica) ---
    for index, row in df_corretti.iterrows():
        codice = str(row['codice'])
        target = row['target_vero']

        # Costruisce il prompt costringendo il modello a prendere una posizione netta
        prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
        messages = [
            {"role": "system", "content": "You are a cybersecurity expert. Your task is to find vulnerabilities in the source code."},
            {"role": "user", "content": prompt}
        ]

        # Applica il template corretto per quel modello (es. i tag [INST] di Llama o <|im_start|> di Qwen)
        testo_formattato = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([testo_formattato], return_tensors="pt").to(model.device)

        with torch.no_grad(): # Disabilita il calcolo dei gradienti (non stiamo addestrando, stiamo solo inferendo)
            # FA LEGGERE IL TESTO. output_hidden_states=True ci fa spiare dentro la rete. 
            # In questo momento, arrivato al layer bersaglio, l'hook SCATTA AUTOMATICAMENTE in background.
            outputs = model(**inputs, output_hidden_states=True)

        # Salta il layer 0 (che spesso è solo l'embedding iniziale delle parole)
        hidden_states = outputs.hidden_states[1:] 

        # Smista le attivazioni layer per layer nei rispettivi scatoloni
        for layer_idx in range(num_layers):
            # Prende il vettore [0, -1, :], ovvero: [batch 0, ULTIMO token generato, tutti i neuroni]
            vettore_ultimo_token = hidden_states[layer_idx][0, -1, :].detach().cpu().numpy()
            
            if target == 'Vulnerabile':
                attivazioni_vulnerabili_steered[layer_idx].append(vettore_ultimo_token)
            elif target == 'Sicuro':
                attivazioni_sicuri_steered[layer_idx].append(vettore_ultimo_token)

    # Il dataset è finito. Togliamo il posto di blocco per riportare il modello alla normalità
    hook_handle.remove()

    # --- 6. CALCOLO DELLA NORMA L2 E GRAFICO ---
    magnitudo_layer_steered = []
    magnitudo_layer_vanilla = [] 

    for layer_idx in range(num_layers):
        # Sicurezza: se uno scatolone è vuoto, metti 0 per evitare crash matematici
        if len(attivazioni_vulnerabili_steered[layer_idx]) == 0 or len(attivazioni_sicuri_steered[layer_idx]) == 0:
            magnitudo_layer_steered.append(0)
            magnitudo_layer_vanilla.append(0)
            continue
            
        # Calcola il punto medio concettuale (centroide) dei codici manipolati
        media_vuln_steered = np.mean(np.stack(attivazioni_vulnerabili_steered[layer_idx]), axis=0)
        media_sicuro_steered = np.mean(np.stack(attivazioni_sicuri_steered[layer_idx]), axis=0)
        
        # Calcola la distanza vettoriale tra i due nuovi concetti
        vettore_steering_steered = media_vuln_steered - media_sicuro_steered
        
        # Applica il Teorema di Pitagora (np.linalg.norm) per trovare la LUNGHEZZA di questa distanza
        magnitudo_layer_steered.append(np.linalg.norm(vettore_steering_steered))

        # Va a ripescare dal disco la lunghezza "Vanilla" calcolata originariamente (per fare il confronto)
        percorso_vettore_vanilla = f"attivazioni_totali/steering_vector_{nome_file_safe}_layer_{layer_idx}.npy"
        if os.path.exists(percorso_vettore_vanilla):
            vec_vanilla = np.load(percorso_vettore_vanilla)
            magnitudo_layer_vanilla.append(np.linalg.norm(vec_vanilla))
        else:
            magnitudo_layer_vanilla.append(0)

    # Avvia la creazione dell'immagine (10x6 pollici)
    plt.figure(figsize=(10, 6))

    # Disegna la linea BLU (La campana intonsa originale)
    if any(magnitudo_layer_vanilla):
        plt.plot(range(num_layers), magnitudo_layer_vanilla, marker='o', linestyle='-', color='blue', label='Vanilla (Sano)', alpha=0.5)

    # Disegna la linea ROSSA (La curva sotto l'effetto dello steering)
    plt.plot(range(num_layers), magnitudo_layer_steered, marker='x', linestyle='-', color='red', label=f'Steered (Iniezione a L{layer_iniezione})', linewidth=2)

    # Disegna una linea verticale per indicare visivamente dove è avvenuta l'iniezione
    plt.axvline(x=layer_iniezione, color='black', linestyle='--', label='Punto di Iniezione Steering')

    # Aggiunge titoli, etichette e griglia di sfondo
    plt.title(f'Schiacciamento della Vulnerabilità sotto Steering - {model_name}')
    plt.xlabel('Indice del Layer')
    plt.ylabel('Magnitudo del Vettore Steering (Norma L2)')
    plt.legend()
    plt.grid(True)

    # Salva il grafico finale su disco
    percorso_grafico = f"{nome_file_safe}_campana_comparativa.png"
    plt.savefig(percorso_grafico)
    plt.close()
    print(f"Grafico salvato in: {percorso_grafico}")

    # --- 7. PULIZIA DELLA SALA OPERATORIA ---
    # Fondamentale per evitare che la VRAM della GPU esploda prima di passare al modello successivo
    try:
        del model, tokenizer, inputs, outputs
        del attivazioni_vulnerabili_steered, attivazioni_sicuri_steered
    except NameError:
        pass
    gc.collect() # Forza il Garbage Collector di Python
    torch.cuda.empty_cache() # Svuota la cache fisica della scheda video Nvidia