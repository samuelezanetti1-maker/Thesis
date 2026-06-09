import os
import pandas as pd
import torch
import gc
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import models_config 

# Ottimizzazione dell'allocazione di memoria CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/scratch_share/bislab/HF_HUB_CACHE/"

# Creiamo cartelle dedicate per non sovrascrivere nulla
os.makedirs("grafici_L2_fail", exist_ok=True)
os.makedirs("CSV tesi/Dati_Grafici_L2", exist_ok=True)

# ==========================================
# 1. CARICAMENTO DATASET RISULTATI
# ==========================================

# A) Caricamento Ground Truth (Baseline fisiologica)
df_baseline = pd.read_csv("CSV tesi/Dataset/dataset_TRUE.csv")

# B) Caricamento dei 3 file dei risultati degli attacchi
# NOTA: Controlla che i percorsi siano esatti (se hai messo AP dentro "Fixed", aggiorna il percorso qui!)
percorso_aa = "CSV tesi/Fixed/risultati_attacco_advanced.csv"
percorso_pi = "CSV tesi/Fixed/risultati_attacco_PJ.csv"
percorso_ap = "CSV tesi/Fixed/risultati_attacco_adversarial_perturbation.csv" # Metti il path corretto

try:
    df_aa = pd.read_csv(percorso_aa)
    df_pi = pd.read_csv(percorso_pi)
    df_ap = pd.read_csv(percorso_ap)
except FileNotFoundError as e:
    print(f"ERRORE CRITICO: Non trovo il file {e.filename}. Controlla i percorsi!")
    exit()

# FILTRO MAGICO: Prendi solo le righe in cui l'attacco HA FALLITO (Modello predice 'Vulnerabile')
df_aa_fail = df_aa[df_aa['target_predetto_adv'] == 'Vulnerabile']
df_pi_fail = df_pi[df_pi['target_predetto_pj'] == 'Vulnerabile']
df_ap_fail = df_ap[df_ap['target_predetto_ap'] == 'Vulnerabile']

for model_name, config in models_config.items():
    print(f"\n{'='*70}\nAnalisi Sismografo L2 - TUTTI I FALLIMENTI: {model_name}\n{'='*70}")
    
    # Filtro i dati sicuri per il modello corrente
    df_corretti_sicuri = df_baseline[(df_baseline['modello'] == model_name) & (df_baseline['target_vero'] == 'Sicuro')]
    
    # Filtro i fallimenti per il modello corrente
    df_aa_modello = df_aa_fail[df_aa_fail['modello'] == model_name]
    df_pi_modello = df_pi_fail[df_pi_fail['modello'] == model_name]
    df_ap_modello = df_ap_fail[df_ap_fail['modello'] == model_name]

    if len(df_corretti_sicuri) == 0:
        print("Nessun dato 'Sicuro' trovato per la baseline. Salto il modello.")
        continue

    nome_file_safe = model_name.replace('/', '_')

    # Inizializzazione LLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        dtype=torch.float16
    )

    num_layers = len(model.model.layers)
    
    # Strutture dati
    attivazioni_sicuri_vanilla = {i: [] for i in range(num_layers)}
    attivazioni_AA_fail = {i: [] for i in range(num_layers)}
    attivazioni_PI_fail = {i: [] for i in range(num_layers)}
    attivazioni_AP_fail = {i: [] for i in range(num_layers)}

    # --- 2. ESTRAZIONE MASSIVA DELLE ATTIVAZIONI ---
    def estrai_vettori(codice_test, dict_attivazioni):
        prompt_template = "Analyze this code \n\nCode:\n{CODE}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
        prompt = prompt_template.replace("{CODE}", str(codice_test))
        messages = [{"role": "system", "content": "You are a cybersecurity expert."}, {"role": "user", "content": prompt}]
        testo_formattato = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([testo_formattato], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        hidden = outputs.hidden_states[1:] 
        for layer_idx in range(num_layers):
            vec = hidden[layer_idx][0, -1, :].detach().cpu().numpy()
            dict_attivazioni[layer_idx].append(vec)
        
        del inputs, outputs, hidden

    # Estrazione Baseline
    print(" -> Estrazione vettori dal Manifold Sicuro...")
    for _, row in df_corretti_sicuri.iterrows():
        estrai_vettori(row['codice'], attivazioni_sicuri_vanilla)

    # Estrazione Attacchi Falliti
    print(f" -> Estrazione Advanced Adversarial (Falliti: {len(df_aa_modello)})...")
    for _, row in df_aa_modello.iterrows():
        estrai_vettori(row['codice_perturbato'], attivazioni_AA_fail)

    print(f" -> Estrazione Prompt Injection (Falliti: {len(df_pi_modello)})...")
    for _, row in df_pi_modello.iterrows():
        estrai_vettori(row['codice_perturbato'], attivazioni_PI_fail)

    print(f" -> Estrazione Adversarial Perturbations (Falliti: {len(df_ap_modello)})...")
    for _, row in df_ap_modello.iterrows():
        estrai_vettori(row['codice_perturbato'], attivazioni_AP_fail)

    # --- 3. CALCOLO GEOMETRICO DELLO SHIFT (NORMA L2) ---
    print(" -> Calcolo delle deviazioni latenti (L2 Norm)...")
    magnitudo_AA = []
    magnitudo_PI = []
    magnitudo_AP = []
    magnitudo_vanilla = [] 

    for layer_idx in range(num_layers):
        media_sicuro = np.mean(np.stack(attivazioni_sicuri_vanilla[layer_idx]), axis=0)

        # AA
        if len(attivazioni_AA_fail[layer_idx]) > 0:
            media_AA = np.mean(np.stack(attivazioni_AA_fail[layer_idx]), axis=0)
            magnitudo_AA.append(np.linalg.norm(media_AA - media_sicuro))
        else:
            magnitudo_AA.append(0)

        # PI
        if len(attivazioni_PI_fail[layer_idx]) > 0:
            media_PI = np.mean(np.stack(attivazioni_PI_fail[layer_idx]), axis=0)
            magnitudo_PI.append(np.linalg.norm(media_PI - media_sicuro))
        else:
            magnitudo_PI.append(0)

        # AP
        if len(attivazioni_AP_fail[layer_idx]) > 0:
            media_AP = np.mean(np.stack(attivazioni_AP_fail[layer_idx]), axis=0)
            magnitudo_AP.append(np.linalg.norm(media_AP - media_sicuro))
        else:
            magnitudo_AP.append(0)

        # Baseline Vanilla (dal disco)
        percorso_vettore_vanilla = f"attivazioni_totali/steering_vector_{nome_file_safe}_layer_{layer_idx}.npy"
        if os.path.exists(percorso_vettore_vanilla):
            vec_vanilla = np.load(percorso_vettore_vanilla)
            magnitudo_vanilla.append(np.linalg.norm(vec_vanilla))
        else:
            magnitudo_vanilla.append(0)

    # --- 4. GENERAZIONE DEI GRAFICI SEPARATI ---
    configurazioni_plot = [
        ("Advanced_Adversarial_FAIL", magnitudo_AA, "red", "Overload: Adv. Adversarial (Fallita)"),
        ("Prompt_Injection_FAIL", magnitudo_PI, "green", "Overload: Prompt Injection (Fallita)"),
        ("Adversarial_Perturbations_FAIL", magnitudo_AP, "orange", "Overload: Adv. Perturbations (Fallita)")
    ]

    print(" -> Generazione dei grafici individuali...")
    for nome_attacco, magnitudo_attacco, colore, label_attacco in configurazioni_plot:
        if sum(magnitudo_attacco) == 0:
            continue # Salta la creazione del grafico se non ci sono dati per questo attacco

        plt.figure(figsize=(10, 6))

        if any(magnitudo_vanilla):
            plt.plot(range(num_layers), magnitudo_vanilla, marker='o', linestyle='--', color='blue', label='Baseline (Vulnerabile Vanilla)', alpha=0.6)

        plt.plot(range(num_layers), magnitudo_attacco, marker='x', linestyle='-', color=colore, label=label_attacco, linewidth=2.5)

        plt.title(f'Impronta L2 (Attacchi Falliti): {nome_attacco.replace("_", " ")}\n{model_name}')
        plt.xlabel('Indice del Layer')
        plt.ylabel('Deviazione dal Manifold Sicuro (Norma L2)')
        plt.legend()
        plt.grid(True)

        percorso_grafico = f"grafici_L2_fail/{nome_file_safe}_{nome_attacco}.png"
        plt.savefig(percorso_grafico)
        plt.close()

    # --- 5. ESPORTAZIONE DATI GREZZI IN CSV ---
    df_export = pd.DataFrame({
        'Layer': range(num_layers),
        'Baseline_Vanilla': magnitudo_vanilla,
        'Advanced_Adversarial_Fallita': magnitudo_AA,
        'Prompt_Injection_Fallita': magnitudo_PI,
        'Adversarial_Perturbation_Fallita': magnitudo_AP
    })
    
    csv_path = f"CSV tesi/Dati_Grafici_L2/{nome_file_safe}_BlackBox_L2_ALL_FAIL.csv"
    df_export.to_csv(csv_path, index=False)
    print(f" -> Dati numerici salvati in: {csv_path}")

    # Pulizia VRAM
    try:
        del model, tokenizer
        del attivazioni_sicuri_vanilla, attivazioni_AA_fail, attivazioni_PI_fail, attivazioni_AP_fail
    except NameError:
        pass
    gc.collect()
    torch.cuda.empty_cache()