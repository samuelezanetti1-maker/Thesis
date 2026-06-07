import os
import pandas as pd
import torch
import gc
import numpy as np
import joblib
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import models_config

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/scratch_share/bislab/HF_HUB_CACHE/"

# --- 1. CONFIGURAZIONE ---
moltiplicatori_per_modello = {
    "Qwen/Qwen2.5-7B-Instruct": 20,
    "Qwen/Qwen2.5-Coder-7B-Instruct": 30,
    "meta-llama/Llama-3.1-8B-Instruct": 3,
    "codellama/CodeLlama-7b-Instruct-hf": 15,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 10,
    "deepseek-ai/deepseek-coder-6.7b-instruct": 8
}

df_TP = pd.read_csv("CSV tesi/Dataset/dataset_TP.csv")
risultati_massivi_diagnostica = []

# Creiamo la cartella per i salvataggi
os.makedirs("CSV tesi/Diagnostica", exist_ok=True)
file_output = "CSV tesi/Diagnostica/risultati_probing_massivo.csv"

# --- 2. HOOK OFFENSIVA ---
def crea_hook_offensiva(vettore_tensore, moltiplicatore):
    vettore_norm = vettore_tensore / torch.norm(vettore_tensore)
    def steering_hook(module, input, output):
        if isinstance(output, tuple):
            tensore = output[0].clone()
        else:
            tensore = output.clone()
        
        vettore_locale = vettore_norm.to(tensore.device)
        
        if len(tensore.shape) == 3:
            tensore[:, -1, :] = tensore[:, -1, :] - (vettore_locale * moltiplicatore)
        else:
            tensore = tensore - (vettore_locale * moltiplicatore)
            
        if isinstance(output, tuple): return (tensore,) + output[1:]
        else: return tensore
    return steering_hook


# --- 3. CICLO MASSIVO SUI MODELLI ---
for model_name, config in models_config.items():
    
    # Prendiamo un campione di snippet Sicuri (TP) per questo modello.
    # Usiamo un sample (es. 50 snippet) per avere solidità statistica ma tempi rapidi.
    df_mod = df_TP[df_TP['modello'] == model_name]
    if len(df_mod) == 0: continue
    

    nome_modello_pulito = model_name.replace("/", "_")
    moltiplicatore = moltiplicatori_per_modello.get(model_name)

    print("\n" + "="*70)
    print(f" DIAGNOSTICA SEMANTICA DI MASSA SU: {model_name}")
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

        num_layers = len(model.model.layers)
        layer_di_attacco = int(num_layers * 0.59) # Il famoso 59%

        percorso_vettore = f"attivazioni_totali/steering_vector_{nome_modello_pulito}_layer_{layer_di_attacco}.npy"
        if not os.path.exists(percorso_vettore):
            print(f"[!] Vettore mancante al layer {layer_di_attacco}. Salto.")
            continue

        vettore_steering = torch.tensor(np.load(percorso_vettore), dtype=model.dtype).to(model.device)

        # Pre-carichiamo tutti i Prober in RAM (velocizza enormemente il ciclo)
        probers = {}
        for layer_idx in range(num_layers):
            percorso_prober = f"probers_salvati/{nome_modello_pulito}/prober_layer_{layer_idx}.pkl"
            if os.path.exists(percorso_prober):
                probers[layer_idx] = joblib.load(percorso_prober)

        print(f"Analisi di {len(df_mod)} snippet in corso...")

        # --- CICLO SUGLI SNIPPET ---
        for index, row in df_mod.iterrows():
            codice = str(row['codice'])
            prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
            messages = [{"role": "system", "content": "You are a cybersecurity expert."}, {"role": "user", "content": prompt}]
            
            testo_formattato = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([testo_formattato], return_tensors="pt").to(model.device)

            with torch.no_grad():
                # FASE A: Modello Vergine (Solo elaborazione del prompt, niente .generate!)
                outputs_vergine = model(**inputs, output_hidden_states=True)
                
                # FASE B: Modello Attaccato
                hook_handle = model.model.layers[layer_di_attacco].register_forward_hook(
                    crea_hook_offensiva(vettore_steering, moltiplicatore)
                )
                outputs_attaccato = model(**inputs, output_hidden_states=True)
                hook_handle.remove()

            # Estrazione delle probabilità layer per layer
            for layer_idx in range(num_layers):
                if layer_idx not in probers: continue # Se manca il prober, saltiamo
                
                prober_corrente = probers[layer_idx]
                
                # Attivazione vergine
                att_vergine = outputs_vergine.hidden_states[layer_idx][0, -1, :].detach().cpu().numpy().reshape(1, -1)
                prob_vergine = prober_corrente.predict_proba(att_vergine)[0][1] * 100
                
                # Attivazione attaccata
                att_attaccata = outputs_attaccato.hidden_states[layer_idx][0, -1, :].detach().cpu().numpy().reshape(1, -1)
                prob_attaccata = prober_corrente.predict_proba(att_attaccata)[0][1] * 100

                risultati_massivi_diagnostica.append({
                    "modello": model_name,
                    "id_snippet": index,
                    "layer": layer_idx,
                    "probabilita_vergine": round(prob_vergine, 2),
                    "probabilita_attaccata": round(prob_attaccata, 2)
                })

            del inputs, outputs_vergine, outputs_attaccato
            torch.cuda.empty_cache()

        # Salvataggio progressivo
        df_temp = pd.DataFrame(risultati_massivi_diagnostica)
        df_temp.to_csv(file_output, index=False)
        print(f"-> Salvataggio progressivo completato.")

        del model, tokenizer, probers
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Errore su {model_name}: {e}")
        continue

print(f"\n[+] DIAGNOSTICA COMPLETATA. Risultati in: {file_output}")