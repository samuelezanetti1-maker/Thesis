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
  


df_TP = pd.read_csv("CSV tesi/Dataset/dataset_TP.csv")
risultati_attacco_steering = []
risultati_aggregati_centrale = []
log_riassunto_asr = []

# --- 2. HOOK OFFENSIVA CON TRACCIAMENTO NORMA L2 ---
def crea_hook_offensiva(vettore_tensore, moltiplicatore, registratore_norme):
    vettore_norm = vettore_tensore / torch.norm(vettore_tensore)

    def steering_hook_offensiva(module, input, output):
        if isinstance(output, tuple):
            tensore_originale = output[0].clone()
        else:
            tensore_originale = output.clone()

        # 1. MISURAZIONE PRE-ATTACCO (Norma L2)
        if len(tensore_originale.shape) == 3:
            norma_pre = torch.norm(tensore_originale[:, -1, :]).item()
        elif len(tensore_originale.shape) == 2:
            norma_pre = torch.norm(tensore_originale).item()
            
        registratore_norme['pre'].append(norma_pre)

        # 2. INIEZIONE DELLO STEERING
        tensore_modificato = tensore_originale.clone()
        vettore_locale = vettore_norm.to(tensore_modificato.device)
        
        if len(tensore_modificato.shape) == 3:
            tensore_modificato[:, -1, :] = tensore_modificato[:, -1, :] - (vettore_locale * moltiplicatore)
            # 3. MISURAZIONE POST-ATTACCO (Prefill)
            norma_post = torch.norm(tensore_modificato[:, -1, :]).item()
        elif len(tensore_modificato.shape) == 2:
            tensore_modificato = tensore_modificato - (vettore_locale * moltiplicatore)
            # 3. MISURAZIONE POST-ATTACCO (Decodifica)
            norma_post = torch.norm(tensore_modificato).item()
            
        registratore_norme['post'].append(norma_post)

        if isinstance(output, tuple):
            return (tensore_modificato,) + output[1:]
        else:
            return tensore_modificato
            
    return steering_hook_offensiva


# --- 3. CICLO DI ATTACCO MASSIVO (Modificato) ---
for model_name, config in models_config.items():
    
    df_mod = df_TP[df_TP['modello'] == model_name]
    if len(df_mod) == 0:
        print(f"\nNessun TP da attaccare per {model_name}. Salto.")
        continue

    print("\n" + "="*60)
    print(f"PREPARAZIONE FAST STEERING SU: {model_name}")
    print("="*60)

    try:
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

        # --- MODIFICA CRUCIALE: IL GOLDEN NUMBER (59%) ---
        numero_layer_totali = len(model.model.layers)
        layer_corretto = int(numero_layer_totali * 0.59) 
        print(f"Il modello ha {numero_layer_totali} layer. Attacchiamo il Locus Causale al 59%: Layer {layer_corretto}")

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
        os.makedirs("txt_tesi/Risposte central_Steering", exist_ok=True)
        percorso_txt_log = f"txt_tesi/Risposte central_Steering/Log_Risposte_Steering_{nome_modello_pulito}.txt"

        with open(percorso_txt_log, "w", encoding="utf-8") as f_log:
            f_log.write(f"=== LOG RISPOSTE STEERING: {model_name} (Layer Locus: {layer_corretto}) ===\n")
            f_log.write("="*60 + "\n\n")

        for moltiplicatore in moltiplicatori_correnti:
            
            with open(percorso_txt_log, "a", encoding="utf-8") as f_log:
                f_log.write(f"\n>>> TEST CON MOLTIPLICATORE: {moltiplicatore} <<<\n\n")
            
            # Creiamo un dizionario per "contrabbandare" le misurazioni fuori dall'hook
            registratore_norme = {'pre': [], 'post': []}
            
            hook_handle = model.model.layers[layer_corretto].register_forward_hook(
                crea_hook_offensiva(vettore_tensore, moltiplicatore, registratore_norme)
            )
            
            attacchi_riusciti = 0
            errori_formattazione = 0

            ### TESTING SUI SINGOLI SNIPPET
            for index, row in df_mod.iterrows():
                # Svuotiamo il registratore prima di ogni snippet
                registratore_norme['pre'].clear()
                registratore_norme['post'].clear()
                
                codice = str(row['codice'])
                prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
                messages = [
                    {"role": "system", "content": "You are a cybersecurity expert. Your task is to find vulnerabilities in the source code."},
                    {"role": "user", "content": prompt}
                ]
                
                testo_formattato = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([testo_formattato], return_tensors="pt").to(model.device)

                token_limit = 1500 if "deepseek" in model_name.lower() else 250

                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=token_limit, 
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                output_ids = generated_ids[0][len(inputs.input_ids[0]):]
                risposta_steered = tokenizer.decode(output_ids, skip_special_tokens=True).replace('Ġ', ' ').replace('Ċ', '\n')

                # Calcoliamo la media delle norme per questo snippet
                media_norma_pre = np.mean(registratore_norme['pre'])
                media_norma_post = np.mean(registratore_norme['post'])
                differenza_norma = media_norma_post - media_norma_pre

                with open(percorso_txt_log, "a", encoding="utf-8") as f_log:
                    f_log.write(f"Snippet ID {index}:\n")
                    f_log.write(f"Norma L2 Pre: {media_norma_pre:.2f} | Post: {media_norma_post:.2f} | Diff: {differenza_norma:.2f}\n")
                    f_log.write(f"{risposta_steered}\n")
                    f_log.write("-" * 50 + "\n\n")
                
                target_predetto_nuovo = evaluate_response(risposta_steered)
                
                if target_predetto_nuovo == "Sicuro":
                    attacchi_riusciti += 1
                elif "FINAL_VERDICT" not in risposta_steered:
                    errori_formattazione += 1
                    
                # Salviamo i dati delle norme nel CSV finale!
                risultati_attacco_steering.append({
                    "id_snippet": index,
                    "modello": model_name,
                    "layer_attaccato": layer_corretto,
                    "moltiplicatore": moltiplicatore,
                    "target_vero": "Vulnerabile",
                    "predizione_baseline": "Vulnerabile",
                    "predizione_post_steering": target_predetto_nuovo,
                    "norma_L2_originale": round(media_norma_pre, 3),
                    "norma_L2_modificata": round(media_norma_post, 3),
                    "delta_L2": round(differenza_norma, 3)
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
df_steering.to_csv("CSV tesi/risultati_attacco_hidden_state_steering.csv", index=False)
print("\nSalvataggio CSV completato in 'CSV tesi/risultati_attacco_hidden_state_steering.csv'")

percorso_txt_riassunto = "CSV tesi/riassunto_ASR_centrale.txt"
with open(percorso_txt_riassunto, "w", encoding="utf-8") as f:
    f.write("=== RISULTATI ABLATION STUDY (ATTACCO LAYER CENTRALE) ===\n")
    f.write("="*60 + "\n\n")
    for riga in log_riassunto_asr:
        f.write(riga + "\n")

df_aggregato_centrale = pd.DataFrame(risultati_aggregati_centrale)
df_aggregato_centrale.to_csv("CSV tesi/percentuali_L2_hidden_state.csv", index=False)
print("CSV aggregato salvato in 'CSV tesi/percentuali_L2_hidden_state.csv'")
