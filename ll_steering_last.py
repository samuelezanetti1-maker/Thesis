import os
import pandas as pd
import torch
import gc
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import models_config

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/scratch_share/bislab/HF_HUB_CACHE/"

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
moltiplicatori_per_modello = {
    "Qwen/Qwen2.5-7B-Instruct": 20,
    "Qwen/Qwen2.5-Coder-7B-Instruct": 30,
    "meta-llama/Llama-3.1-8B-Instruct": 3,
    "codellama/CodeLlama-7b-Instruct-hf": 15,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 10,
    "deepseek-ai/deepseek-coder-6.7b-instruct": 8
}

layer_per_modello = {
    "Qwen/Qwen2.5-7B-Instruct": 26,
    "Qwen/Qwen2.5-Coder-7B-Instruct": 26,
    "meta-llama/Llama-3.1-8B-Instruct": 30,
    "codellama/CodeLlama-7b-Instruct-hf": 30,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 26,
    "deepseek-ai/deepseek-coder-6.7b-instruct": 30
}

N_CAMPIONI_MAX = 900

# ==========================================
# 2. CARICAMENTO DATI (Solo il CSV dei Risultati!)
# ==========================================
print("Caricamento dataset dei successi...")
df_steering_results = pd.read_csv("CSV tesi/Fixed/risultati_attacco_steering.csv")
os.makedirs("txt_tesi/Logit_Lens_Targeted", exist_ok=True)

# ==========================================
# 3. HOOK DI DIFESA (Steering)
# ==========================================
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

# ==========================================
# 4. CICLO SUI MODELLI
# ==========================================
for model_name, config in models_config.items():
    print(f"\n{'='*70}\n ANALISI CHIRURGICA STEERING: {model_name}\n{'='*70}")
    
    moltiplicatore_target = moltiplicatori_per_modello.get(model_name)
    
    # Filtriamo i successi per questo modello
    df_successi = df_steering_results[
        (df_steering_results['modello'] == model_name) & 
        (df_steering_results['predizione_post_steering'] == 'Sicuro') &
        (df_steering_results['moltiplicatore'] == moltiplicatore_target)
    ]
    
    if len(df_successi) == 0:
        print(f" [!] Nessun successo di steering trovato per {model_name}. Salto.")
        continue
        
    # ESTREZIONE DIRETTA DAL CSV DEI RISULTATI
    codici_da_testare = df_successi['codice_originale'].tolist()[:N_CAMPIONI_MAX]
    
    print(f" -> Trovati {len(codici_da_testare)} prompt di successo. Inizio estrazione...")

    nome_file_safe = model_name.replace('/', '_')
    layer_locus = layer_per_modello.get(model_name)

    path_steering = f"attivazioni_totali/steering_vector_{nome_file_safe}_layer_{layer_locus}.npy"
    if not os.path.exists(path_steering):
        print(f" [!] Vettore steering mancante. Salto il modello.")
        continue

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            low_cpu_mem_usage=True,
            dtype=torch.float16
        )

        layer_penultimo = len(model.model.layers) - 1 

        vettore_numpy = np.load(path_steering)
        vettore_tensore = torch.tensor(vettore_numpy, dtype=model.dtype, device=model.device)

        shift_locus_list = []
        shift_penultimo_list = []

        # Estrazione con e senza hook per ogni codice
        for codice in codici_da_testare:
            prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
            messages = [
                {"role": "system", "content": "You are a cybersecurity expert. Your task is to find vulnerabilities in the source code."},
                {"role": "user", "content": prompt}
            ]
            testo = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([testo], return_tensors="pt").to(model.device)
            
            # --- 1. RUN BASELINE (Senza Difesa) ---
            with torch.no_grad():
                out_base = model(**inputs, output_hidden_states=True)
            hidden_base_locus = out_base.hidden_states[layer_locus][0, -1, :]
            hidden_base_penultimo = out_base.hidden_states[layer_penultimo][0, -1, :]
            
            # --- 2. RUN STEERED (Con Difesa Attiva) ---
            hook_handle = model.model.layers[layer_locus].register_forward_hook(
                crea_hook_offensiva(vettore_tensore, moltiplicatore_target)
            )
            
            with torch.no_grad():
                out_steered = model(**inputs, output_hidden_states=True)
                
            hook_handle.remove() # Pulizia hook
            
            hidden_steered_locus = out_steered.hidden_states[layer_locus][0, -1, :]
            hidden_steered_penultimo = out_steered.hidden_states[layer_penultimo][0, -1, :]
            
            # --- 3. CALCOLO DEL DELTA ---
            shift_locus_list.append(hidden_steered_locus - hidden_base_locus)
            shift_penultimo_list.append(hidden_steered_penultimo - hidden_base_penultimo)

        # Medie vettoriali dello shift difensivo
        vettore_difesa_locus = torch.mean(torch.stack(shift_locus_list), dim=0)
        vettore_difesa_penultimo = torch.mean(torch.stack(shift_penultimo_list), dim=0)

        # ==========================================
        # 5. LOGIT LENS 
        # ==========================================
        lm_head = model.get_output_embeddings() 
        final_layernorm = model.model.norm  

        id_true = tokenizer.encode(" True", add_special_tokens=False)[-1]
        id_false = tokenizer.encode(" False", add_special_tokens=False)[-1]
        w_true = lm_head.weight[id_true]
        w_false = lm_head.weight[id_false]

        report_path = f"txt_tesi/Logit_Lens_Targeted/Steering_Impact_{nome_file_safe}_last.txt"

        def decodifica_contrastiva(vettore_shift, nome_layer, file_log):
            with torch.no_grad():
                v_calc = vettore_shift.to(model.dtype)
                
                # Applichiamo i pesi della RMSNorm
                v_calc_scaled = v_calc * final_layernorm.weight
                
                logit_true = torch.dot(v_calc_scaled, w_true).item()
                logit_false = torch.dot(v_calc_scaled, w_false).item()
                delta = logit_true - logit_false
            
            file_log.write(f"--- IMPATTO DIFESA AL LAYER: {nome_layer} ---\n")
            file_log.write(f"  -> Spinta causata su ' True' (Vulnerabile) : {logit_true:.4f}\n")
            file_log.write(f"  -> Spinta causata su ' False' (Sicuro)     : {logit_false:.4f}\n")
            file_log.write(f"  -> DELTA PUSH (True - False)               : {delta:.4f}\n")
            
            if delta < -1.0:
                file_log.write("  [Analisi] -> DIFESA ATTIVA: Lo steering sta spostando fortemente la semantica verso 'Sicuro'.\n\n")
            elif delta > 1.0:
                file_log.write("  [Analisi] -> PARADOSSO: Lo steering sta spingendo verso la vulnerabilità.\n\n")
            else:
                file_log.write("  [Analisi] -> ASSORBIMENTO/PLANE SHIFT: Il modello ha riassorbito la spinta o l'ha deviata.\n\n")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"REPORT TARGETED STEERING - {model_name}\n")
            f.write(f"Analisi condotta su {len(codici_da_testare)} prompt protetti con successo.\n")
            f.write(f"Layer Iniezione: {layer_locus} | Layer Penultimo: {layer_penultimo}\n")
            f.write("Calcolo del vettore 'Azione di Difesa' (Attivazione Steerata - Attivazione Baseline).\n\n")
            
            decodifica_contrastiva(vettore_difesa_locus, f"Layer Iniezione (Locus {layer_locus})", f)
            decodifica_contrastiva(vettore_difesa_penultimo, f"Layer Penultimo ({layer_penultimo})", f)
        
        print(f" [+] Report completato e salvato in: {report_path}")

        del model, tokenizer, vettore_tensore, shift_locus_list, shift_penultimo_list
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Errore su {model_name}: {e}")

print("\nFinito! Tutti i report chirurgici sono pronti.")