from config import models_config
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/scratch_share/bislab/HF_HUB_CACHE/"

import pandas as pd
import torch
import gc
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import evaluate_response

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
  
os.makedirs("controprova/txt_tesi", exist_ok=True)
os.makedirs("controprova/CSV tesi", exist_ok=True)

df_TP = pd.read_csv("CSV tesi/Split_Dataset_succ/steering_solo_successi.csv")
risultati_attacco_steering = []
risultati_aggregati_centrale = []
log_riassunto_asr = []

# --- 2. HOOK OFFENSIVA ---
def crea_hook_offensiva(vettore_tensore, moltiplicatore):
    vettore_norm = vettore_tensore / torch.norm(vettore_tensore)

    def steering_hook_offensiva(module, input, output):
        if isinstance(output, tuple):
            tensore_modificato = output[0].clone()
        else:
            tensore_modificato = output.clone()

        vettore_locale = vettore_norm.to(tensore_modificato.device)
        
        if len(tensore_modificato.shape) == 3:
            tensore_modificato[:, -1, :] = tensore_modificato[:, -1, :] - (vettore_locale * moltiplicatore)
        elif len(tensore_modificato.shape) == 2:
            tensore_modificato = tensore_modificato - (vettore_locale * moltiplicatore)
            
        if isinstance(output, tuple):
            return (tensore_modificato,) + output[1:]
        else:
            return tensore_modificato
    return steering_hook_offensiva

# --- 3. CICLO DI ATTACCO MASSIVO ---
for model_name, config in models_config.items():
    
    df_mod = df_TP[df_TP['modello'] == model_name]
    if len(df_mod) == 0:
        print(f"\nNessun TP da attaccare per {model_name}. Salto.")
        continue

    print("\n" + "="*60)
    print(f"PREPARAZIONE FAST STEERING SU: {model_name}")
    print("="*60)

    try:
        # Recuperiamo la lista dei moltiplicatori specifici per QUESTO modello
        moltiplicatori_correnti = moltiplicatori_per_modello.get(model_name)
        print(f"Userò questi moltiplicatori: {moltiplicatori_correnti}")

        print("Caricamento modello e tokenizer in corso...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=config.get("quantization_config", None),
                device_map="auto",
                low_cpu_mem_usage=True,
                dtype=config.get("dtype", torch.float16)
            )

        layer_corretto =  layer_per_modello.get(model_name)[0]
        print(f"Il modello ha {len(model.model.layers)} layer. Attacchiamo il centrale: {layer_corretto}")

        nome_file_safe = model_name.replace('/', '_')
        percorso_vettore = f"attivazioni_totali/steering_vector_{nome_file_safe}_layer_{layer_corretto}.npy"

        if not os.path.exists(percorso_vettore):
            print(f"[!] Vettore NON TROVATO: {percorso_vettore}")
            print(f"Salto il modello {model_name}.")
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            continue

        vettore_numpy = np.load(percorso_vettore)
        vettore_tensore = torch.tensor(vettore_numpy, dtype=model.dtype, device=model.device)

        nome_modello_pulito = model_name.replace("/", "_")
        os.makedirs("controprova/txt_tesi/Risposte_Steering", exist_ok=True)
        percorso_txt_log = f"controprova/txt_tesi/Risposte_Steering/Log_Risposte_Steering_{nome_modello_pulito}.txt"

        with open(percorso_txt_log, "w", encoding="utf-8") as f_log:
            f_log.write(f"=== LOG RISPOSTE STEERING: {model_name} (Layer Centrale: {layer_corretto}) ===\n")
            f_log.write("="*60 + "\n\n")

        # --- INIZIO CICLO DEI MOLTIPLICATORI (usando quelli specifici!) ---
        for moltiplicatore in moltiplicatori_correnti:
            
            with open(percorso_txt_log, "a", encoding="utf-8") as f_log:
                f_log.write(f"\n>>> TEST CON MOLTIPLICATORE: {moltiplicatore} <<<\n\n")
                
            hook_handle = model.model.layers[layer_corretto].register_forward_hook(
                crea_hook_offensiva(vettore_tensore, moltiplicatore)
            )
            
            attacchi_riusciti = 0
            errori_formattazione = 0

            ### TESTING SUI SINGOLI SNIPPET
            for index, row in df_mod.iterrows():
                codice = str(row['codice'])

                prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with a brief summary of a short sentence followed by: 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure)"
                messages = [
                    {"role": "system", "content": "You are a cybersecurity expert. Your task is to find vulnerabilities in the source code."},
                    {"role": "user", "content": prompt}
                ]
                
                testo_formattato = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([testo_formattato], return_tensors="pt").to(model.device)

                token_limit = 1500 if "deepseek" in model_name.lower() else 500

                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=token_limit, 
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                output_ids = generated_ids[0][len(inputs.input_ids[0]):]
                risposta_steered = tokenizer.decode(output_ids, skip_special_tokens=True)

                risposta_steered = risposta_steered.replace('Ġ', ' ').replace('Ċ', '\n')

                with open(percorso_txt_log, "a", encoding="utf-8") as f_log:
                    f_log.write(f"Snippet ID {index}:\n")
                    f_log.write(f"{risposta_steered}\n")
                    f_log.write("-" * 50 + "\n\n")
                
                target_predetto_nuovo = evaluate_response(risposta_steered)
                
                if target_predetto_nuovo == "Sicuro":
                    attacchi_riusciti += 1
                elif "FINAL_VERDICT" not in risposta_steered:
                    errori_formattazione += 1
                    
                risultati_attacco_steering.append({
                    "id_snippet": index,
                    "modello": model_name,
                    "layer_attaccato": layer_corretto,
                    "moltiplicatore": moltiplicatore,
                    "target_vero": "Vulnerabile",
                    "predizione_baseline": "Vulnerabile",
                    "predizione_post_steering": target_predetto_nuovo
                })

            # Metriche per questo moltiplicatore
            asr_steering = (attacchi_riusciti / len(df_mod)) * 100
            rateo_errori = (errori_formattazione / len(df_mod)) * 100
            
            stringa_log = f"Moltiplicatore: {moltiplicatore:2d} | ASR: {asr_steering:5.2f}% | Errori: {rateo_errori:5.2f}%"
            log_riassunto_asr.append(f"Modello: {model_name} | Layer: {layer_corretto} | {stringa_log}")
            
            print(f"   => {stringa_log}")
            risultati_aggregati_centrale.append({
                "modello": model_name,
                "layer": layer_corretto,
                "moltiplicatore": moltiplicatore,
                "asr_percentuale": asr_steering,
                "errori_percentuale": rateo_errori
            })

            # Pulizia post-moltiplicatore
            hook_handle.remove()
            try:
                del inputs, generated_ids
            except NameError: pass
            torch.cuda.empty_cache()
            
        # --- FINE MODELLO, PULIZIA ---
        try:
            del model, tokenizer
        except NameError: pass
        gc.collect()
        torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Errore durante l'attacco steering su {model_name}: {e}")
        continue

# --- 4. SALVATAGGI FINALI ---
df_steering = pd.DataFrame(risultati_attacco_steering)
df_steering.to_csv("controprova/CSV tesi/risultati_attacco_steering.csv", index=False)
print("\nSalvataggio CSV completato in 'controprova/CSV tesi/risultati_attacco_steering.csv'")

percorso_txt_riassunto = "controprova/txt_tesi/riassunto_ASR.txt"
with open(percorso_txt_riassunto, "w", encoding="utf-8") as f:
    f.write("=== RISULTATI ABLATION STUDY (ATTACCO LAYER CENTRALE) ===\n")
    f.write("="*60 + "\n\n")
    for riga in log_riassunto_asr:
        f.write(riga + "\n")

df_aggregato_centrale = pd.DataFrame(risultati_aggregati_centrale)
df_aggregato_centrale.to_csv("controprova/CSV tesi/percentuali_steering.csv", index=False)
print("CSV aggregato salvato in 'controprova/CSV tesi/percentuali_steering.csv'")
