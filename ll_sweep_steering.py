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
    "Qwen/Qwen2.5-7B-Instruct": 18,
    "Qwen/Qwen2.5-Coder-7B-Instruct": 18,
    "meta-llama/Llama-3.1-8B-Instruct": 15,
    "codellama/CodeLlama-7b-Instruct-hf": 13,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 19,
    "deepseek-ai/deepseek-coder-6.7b-instruct": 16
}


# ==========================================
# 2. CARICAMENTO DATI
# ==========================================
print("Caricamento dataset dei successi...")
# Usa il percorso corretto per il tuo CSV dei risultati
df_steering_results = pd.read_csv("CSV tesi/Fixed/risultati_attacco_steering.csv")
os.makedirs("CSV tesi/Logit_Lens", exist_ok=True)

# Lista globale per il CSV finale
risultati_sweep_difesa = []

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
            tensore_modificato = tensore_modificato - (vettore_locale * moltiplicatore)
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
    print(f"\n{'='*70}\n SWEEP PROPAGAZIONE STEERING: {model_name}\n{'='*70}")
    
    moltiplicatore_target = moltiplicatori_per_modello.get(model_name)
    layer_locus = layer_per_modello.get(model_name)
    nome_file_safe = model_name.replace('/', '_')
    
    df_fallimenti = df_steering_results[
        (df_steering_results['modello'] == model_name) & 
        (df_steering_results['predizione_post_steering'] == 'Sicuro') & # Modificato qui
        (df_steering_results['moltiplicatore'] == moltiplicatore_target)
    ]
    
    if len(df_fallimenti) == 0:
        print(f" [!] Nessun fallimento trovato per {model_name}. Salto.")
        continue
        
    codici_da_testare = df_fallimenti['codice_originale'].tolist()
    print(f" -> Trovati {len(codici_da_testare)} fallimenti. Inizio estrazione...")
    
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

        vettore_numpy = np.load(path_steering)
        vettore_tensore = torch.tensor(vettore_numpy, dtype=model.dtype, device=model.device)

        stati_steered_list = []
        base_states_list = []

        # -- FUNZIONE 1: Estrai tutti i layer BASE (Senza Difesa) --
        def get_all_hidden_states(codice):
            prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
            messages = [
            {"role": "system", "content": "You are a cybersecurity expert. Your task is to find vulnerabilities in the source code."},
            {"role": "user", "content": prompt}
            ]
            
            testo_base = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # --- LOGICA 2-STEP PER MODELLI REASONING ---
            if "DeepSeek-R1" in model_name:
                inputs_gen = tokenizer([testo_base], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs_gen,
                        max_new_tokens=600, # Lasciamo generare la CoT
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=False
                    )
                # Estraiamo solo il pensiero
                token_generati = output_ids[0][inputs_gen.input_ids.shape[1]:]
                testo_generato = tokenizer.decode(token_generati, skip_special_tokens=False)
                
                # Tronchiamo alla fine del pensiero
                if "</think>" in testo_generato:
                    pensiero_puro = testo_generato.split("</think>")[0] + "</think>\n"
                else:
                    pensiero_puro = testo_generato
                
                testo_finale = testo_base + pensiero_puro + "FINAL_VERDICT:"
            else:
                # Per i modelli normali, facciamo Fast-Forward diretto
                testo_finale = testo_base + "FINAL_VERDICT:"
            # -------------------------------------------

            inputs = tokenizer([testo_finale], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            return torch.stack([layer_state[0, -1, :] for layer_state in out.hidden_states])

        # -- FUNZIONE 2: Estrai tutti i layer STEERED (Con Difesa) --
        def get_all_hidden_states_steered(codice):
            prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
            messages = [
            {"role": "system", "content": "You are a cybersecurity expert. Your task is to find vulnerabilities in the source code."},
            {"role": "user", "content": prompt}
            ]
            
            testo_base = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # ATTACCHIAMO L'HOOK SUBITO, PRIMA DI GENERARE LA CoT
            hook_handle = model.model.layers[layer_locus].register_forward_hook(
                crea_hook_offensiva(vettore_tensore, moltiplicatore_target)
            )

            # --- LOGICA 2-STEP PER MODELLI REASONING ---
            if "DeepSeek-R1" in model_name:
                inputs_gen = tokenizer([testo_base], return_tensors="pt").to(model.device)
                
                # Il modello PENSA SOTTO ATTACCO
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs_gen,
                        max_new_tokens=600,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=False
                    )
                token_generati = output_ids[0][inputs_gen.input_ids.shape[1]:]
                testo_generato = tokenizer.decode(token_generati, skip_special_tokens=False)
                
                if "</think>" in testo_generato:
                    pensiero_puro = testo_generato.split("</think>")[0] + "</think>\n"
                else:
                    pensiero_puro = testo_generato
                
                testo_finale = testo_base + pensiero_puro + "FINAL_VERDICT:"
            else:
                testo_finale = testo_base + "FINAL_VERDICT:"
            # -------------------------------------------

            inputs = tokenizer([testo_finale], return_tensors="pt").to(model.device)
            
            # Forward pass finale per estrarre gli stati nascosti (l'hook è ancora attivo)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                
            # RIMUOVIAMO L'HOOK ALLA FINE
            hook_handle.remove()
            return torch.stack([layer_state[0, -1, :] for layer_state in out.hidden_states])

        print(f" -> Esecuzione Forward Pass Multi-Layer...")
        stati_steered_list = []
        
        for codice in codici_da_testare:
            stati_base = get_all_hidden_states(codice)
            stati_steered = get_all_hidden_states_steered(codice)
            
            # Salviamo gli stati assoluti, NON la differenza
            stati_steered_list.append(stati_steered)
            base_states_list.append(stati_base)

        # Media vettoriale: [Num_Layers, Hidden_Dim]
        vettore_steered_all_layers = torch.mean(torch.stack(stati_steered_list), dim=0)
        vettore_base_all_layers = torch.mean(torch.stack(base_states_list), dim=0)

        # ==========================================
        # 5. LOGIT LENS SU TUTTI I LAYER
        # ==========================================
        lm_head = model.get_output_embeddings() 
        final_layernorm = model.model.norm  

        id_true = tokenizer.encode(" True", add_special_tokens=False)[-1]
        id_false = tokenizer.encode(" False", add_special_tokens=False)[-1]
        w_true = lm_head.weight[id_true]
        w_false = lm_head.weight[id_false]

        num_layers_totali = vettore_steered_all_layers.shape[0]
        print(f" -> Mappatura Logit Lens su {num_layers_totali} layer...")

        for layer_idx in range(num_layers_totali):
            # 1. CALCOLO ASSOLUTO STATO BASE
            v_base = vettore_base_all_layers[layer_idx].to(model.dtype)
            v_base_norm = final_layernorm(v_base)
            logit_true_base = torch.dot(v_base_norm, w_true).item()
            logit_false_base = torch.dot(v_base_norm, w_false).item()
            
            # Valore Logit assoluto della rete sana (True - False)
            assoluto_base = logit_true_base - logit_false_base

            # 2. CALCOLO ASSOLUTO STATO STEERED
            v_steered = vettore_steered_all_layers[layer_idx].to(model.dtype)
            v_steered_norm = final_layernorm(v_steered)
            logit_true_steered = torch.dot(v_steered_norm, w_true).item()
            logit_false_steered = torch.dot(v_steered_norm, w_false).item()
            
            # Valore Logit assoluto della rete attaccata (True - False)
            assoluto_steered = logit_true_steered - logit_false_steered

            # 3. CALCOLO DEL PURO SHIFT (Il vero Delta)
            puro_shift = assoluto_steered - assoluto_base
            
            risultati_sweep_difesa.append({
                "MODELLO": model_name,
                "LAYER": layer_idx,
                "Locus_Iniezione": layer_locus,
                "Base_Steer": assoluto_base,    # <-- Nominato già per il tuo plotting!
                "Delta_Steer": puro_shift       # <-- ORA È IL PURO SHIFT!
            })

        print(" [+] Salvataggio dati completato.")


        del model, tokenizer, vettore_tensore, stati_steered_list, base_states_list, vettore_steered_all_layers, vettore_base_all_layers
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Errore su {model_name}: {e}")

# ==========================================
# 6. SALVATAGGIO IN CSV FINALE
# ==========================================
percorso_csv_finale = "CSV tesi/Logit_Lens/Risultati_Sweep_Steering.csv"
df_risultati = pd.DataFrame(risultati_sweep_difesa)
df_risultati.to_csv(percorso_csv_finale, index=False)

print(f"\n{'='*70}\n ELABORAZIONE FINITA!")
print(f" Il dataset completo della propagazione dello Steering è in: {percorso_csv_finale}\n{'='*70}")