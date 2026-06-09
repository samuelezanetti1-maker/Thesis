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

# =====================================================================
# --- 1. CONFIGURAZIONE PROBING STEERING ---
# =====================================================================

# 3 layer con l'accuratezza più alta 
top_probing_layers = {
    "Qwen/Qwen2.5-7B-Instruct": [20],
    "Qwen/Qwen2.5-Coder-7B-Instruct": [20],
    "meta-llama/Llama-3.1-8B-Instruct": [31],
    "codellama/CodeLlama-7b-Instruct-hf": [29],
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": [8],
    "deepseek-ai/deepseek-coder-6.7b-instruct": [31]
}

# I moltiplicatori specifici 
moltiplicatori_per_modello = {
    "Qwen/Qwen2.5-7B-Instruct": [20],
    "Qwen/Qwen2.5-Coder-7B-Instruct": [30],
    "meta-llama/Llama-3.1-8B-Instruct": [3],
    "codellama/CodeLlama-7b-Instruct-hf": [15],
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": [10],
    "deepseek-ai/deepseek-coder-6.7b-instruct": [8]
}
moltiplicatori_default = [5, 10]

# =====================================================================

df_TP = pd.read_csv("CSV tesi/Dataset/dataset_TP.csv")
risultati_attacco_probing = []
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

# --- 3. CICLO DI ATTACCO MULTI-LAYER ---
for model_name, layers_da_attaccare in top_probing_layers.items():
    if model_name not in models_config:
        continue
        
    config = models_config[model_name]
    df_mod = df_TP[df_TP['modello'] == model_name]
    if len(df_mod) == 0:
        continue

    print("\n" + "="*60)
    print(f"PREPARAZIONE PROBING STEERING SU: {model_name}")
    print(f"Layer programmati per l'attacco: {layers_da_attaccare}")
    print("="*60)

    try:
        moltiplicatori_correnti = moltiplicatori_per_modello.get(model_name, moltiplicatori_default)
        print("Caricamento modello e tokenizer...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=config.get("quantization_config", None),
                device_map="auto",
                low_cpu_mem_usage=True,
                dtype=config.get("dtype", torch.float16)
            )

        nome_modello_pulito = model_name.replace("/", "_")
        os.makedirs("txt_tesi/Risposte Probing_Steering", exist_ok=True)
        
        # --- CICLO SUI 3 LAYER MIGLIORI ---
        for layer_corretto in layers_da_attaccare:
            print(f"\n---> BERSAGLIO ATTUALE: LAYER {layer_corretto} <---")
            
            percorso_vettore = f"attivazioni_totali/steering_vector_{nome_modello_pulito}_layer_{layer_corretto}.npy"

            if not os.path.exists(percorso_vettore):
                print(f"[!] Vettore NON TROVATO: {percorso_vettore}")
                continue

            vettore_numpy = np.load(percorso_vettore)
            vettore_tensore = torch.tensor(vettore_numpy, dtype=model.dtype, device=model.device)

            percorso_txt_log = f"txt_tesi/Risposte Probing_Steering/Log_{nome_modello_pulito}_L{layer_corretto}.txt"
            with open(percorso_txt_log, "w", encoding="utf-8") as f_log:
                f_log.write(f"=== LOG PROBING STEERING: {model_name} (Layer Top-Probing: {layer_corretto}) ===\n")
                f_log.write("="*60 + "\n\n")

            # --- CICLO SUI MOLTIPLICATORI ---
            for moltiplicatore in moltiplicatori_correnti:
                with open(percorso_txt_log, "a", encoding="utf-8") as f_log:
                    f_log.write(f"\n>>> TEST CON MOLTIPLICATORE: {moltiplicatore} <<<\n\n")
                    
                hook_handle = model.model.layers[layer_corretto].register_forward_hook(
                    crea_hook_offensiva(vettore_tensore, moltiplicatore)
                )
                
                attacchi_riusciti = 0
                errori_formattazione = 0

                for index, row in df_mod.iterrows():
                    codice = str(row['codice'])
                    prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
                    messages = [
                        {"role": "system", "content": "You are a cybersecurity expert. Your task is to find vulnerabilities in the source code."},
                        {"role": "user", "content": prompt}
                    ]
                    
                    testo_formattato = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer([testo_formattato], return_tensors="pt").to(model.device)
                    token_limit = 1500 if "deepseek" in model_name.lower() else 250

                    generated_ids = model.generate(**inputs, max_new_tokens=token_limit, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                    output_ids = generated_ids[0][len(inputs.input_ids[0]):]
                    risposta_steered = tokenizer.decode(output_ids, skip_special_tokens=True).replace('Ġ', ' ').replace('Ċ', '\n')

                    with open(percorso_txt_log, "a", encoding="utf-8") as f_log:
                        f_log.write(f"Snippet ID {index}:\n{risposta_steered}\n")
                        f_log.write("-" * 50 + "\n\n")
                    
                    target_predetto_nuovo = evaluate_response(risposta_steered)
                    
                    if target_predetto_nuovo == "Sicuro":
                        attacchi_riusciti += 1
                    elif "FINAL_VERDICT" not in risposta_steered:
                        errori_formattazione += 1
                        
                    risultati_attacco_probing.append({
                        "id_snippet": index, "modello": model_name,
                        "layer_attaccato": layer_corretto, "metodo_scelta": "Top-Probing",
                        "moltiplicatore": moltiplicatore, "target_vero": "Vulnerabile",
                        "predizione_post_steering": target_predetto_nuovo
                    })

                asr_steering = (attacchi_riusciti / len(df_mod)) * 100
                rateo_errori = (errori_formattazione / len(df_mod)) * 100
                
                stringa_log = f"L{layer_corretto} | Mult: {moltiplicatore:2d} | ASR: {asr_steering:5.2f}% | Errori: {rateo_errori:5.2f}%"
                log_riassunto_asr.append(f"{model_name} | {stringa_log}")
                print(f"   => {stringa_log}")

                hook_handle.remove()
                try: del inputs, generated_ids
                except NameError: pass
                torch.cuda.empty_cache()
                
        # --- FINE MODELLO ---
        try: del model, tokenizer
        except NameError: pass
        gc.collect()
        torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Errore durante l'attacco steering su {model_name}: {e}")
        continue

# --- 4. SALVATAGGI FINALI ---
os.makedirs("CSV tesi", exist_ok=True)
df_steering = pd.DataFrame(risultati_attacco_probing)
df_steering.to_csv("CSV tesi/risultati_attacco_PROBING_steering.csv", index=False)
print("\nSalvataggio completato in 'CSV tesi/risultati_attacco_PROBING_steering.csv'")

with open("CSV tesi/riassunto_ASR_PROBING.txt", "w", encoding="utf-8") as f:
    f.write("=== RISULTATI ABLATION STUDY (ATTACCO SUI TOP 3 LAYER DA PROBING) ===\n")
    f.write("="*60 + "\n\n")
    for riga in log_riassunto_asr:
        f.write(riga + "\n")