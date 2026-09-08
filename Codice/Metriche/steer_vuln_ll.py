import os
import pandas as pd
import torch
import gc
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import models_config

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 
os.environ["HF_HOME"] = 

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


print("Caricamento dataset dei successi...")
df_steering_results = pd.read_csv("CSV tesi/Fixed/risultati_attacco_steering.csv")
os.makedirs("CSV tesi/Logit_Lens", exist_ok=True)

risultati_sweep_difesa = []


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

# 4. CICLO 
for model_name, config in models_config.items():
    print(f"\n{'='*70}\n SWEEP PROPAGAZIONE STEERING: {model_name}\n{'='*70}")
    
    moltiplicatore_target = moltiplicatori_per_modello.get(model_name)
    layer_locus = layer_per_modello.get(model_name)
    nome_file_safe = model_name.replace('/', '_')
    
    # Filtro: Modello ingannato dall'attacco
    df_fallimenti = df_steering_results[
        (df_steering_results['modello'] == model_name) & 
        (df_steering_results['predizione_post_steering'] == 'Vulnerabile') & 
        (df_steering_results['moltiplicatore'] == moltiplicatore_target)
    ]
    
    if len(df_fallimenti) == 0:
        print(f" [!] Nessun dato trovato per {model_name}. Salto.")
        continue
        
    codici_da_testare = df_fallimenti['codice_originale'].tolist()
    print(f" -> Trovati {len(codici_da_testare)} snippet. Inizio estrazione...")
    
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

        # Inizializzazione pesi Logit Lens
        lm_head = model.get_output_embeddings() 
        final_layernorm = model.model.norm  
        id_true = tokenizer.encode(" True", add_special_tokens=False)[-1]
        id_false = tokenizer.encode(" False", add_special_tokens=False)[-1]
        w_true = lm_head.weight[id_true]
        w_false = lm_head.weight[id_false]

        def estrai_logits_base(codice):
            prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
            messages = [
                {"role": "system", "content": "You are a cybersecurity expert. Your task is to find vulnerabilities in the source code."},
                {"role": "user", "content": prompt}
            ]
            testo_base = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if "DeepSeek-R1" in model_name:
                inputs_gen = tokenizer([testo_base], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(**inputs_gen, max_new_tokens=600, pad_token_id=tokenizer.eos_token_id, do_sample=False)
                token_generati = output_ids[0][inputs_gen.input_ids.shape[1]:]
                testo_generato = tokenizer.decode(token_generati, skip_special_tokens=False)
                pensiero_puro = testo_generato.split("</think>")[0] + "</think>\n" if "</think>" in testo_generato else testo_generato
                testo_finale = testo_base + pensiero_puro + "FINAL_VERDICT:"
            else:
                testo_finale = testo_base + "FINAL_VERDICT:"

            inputs = tokenizer([testo_finale], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            
            # Applichiamo RMSNorm 
            layer_logits = []
            for layer_state in out.hidden_states:
                v_calc = layer_state[0, -1, :].to(model.dtype)
                v_calc_norm = final_layernorm(v_calc)
                logit_true = torch.dot(v_calc_norm, w_true).item()
                logit_false = torch.dot(v_calc_norm, w_false).item()
                layer_logits.append(logit_true - logit_false)
            return layer_logits

        def estrai_logits_steered(codice):
            prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
            messages = [
                {"role": "system", "content": "You are a cybersecurity expert. Your task is to find vulnerabilities in the source code."},
                {"role": "user", "content": prompt}
            ]
            testo_base = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Hook in Generazione
            hook_handle = model.model.layers[layer_locus].register_forward_hook(
                crea_hook_offensiva(vettore_tensore, moltiplicatore_target)
            )

            if "DeepSeek-R1" in model_name:
                inputs_gen = tokenizer([testo_base], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(**inputs_gen, max_new_tokens=600, pad_token_id=tokenizer.eos_token_id, do_sample=False)
                token_generati = output_ids[0][inputs_gen.input_ids.shape[1]:]
                testo_generato = tokenizer.decode(token_generati, skip_special_tokens=False)
                pensiero_puro = testo_generato.split("</think>")[0] + "</think>\n" if "</think>" in testo_generato else testo_generato
                testo_finale = testo_base + pensiero_puro + "FINAL_VERDICT:"
            else:
                testo_finale = testo_base + "FINAL_VERDICT:"

            inputs = tokenizer([testo_finale], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                
            hook_handle.remove()
            
            layer_logits = []
            for layer_state in out.hidden_states:
                v_calc = layer_state[0, -1, :].to(model.dtype)
                v_calc_norm = final_layernorm(v_calc)
                logit_true = torch.dot(v_calc_norm, w_true).item()
                logit_false = torch.dot(v_calc_norm, w_false).item()
                layer_logits.append(logit_true - logit_false)
            return layer_logits

        print(f" -> Esecuzione Forward Pass ed Estrazione Puntuale...")
        
        logits_base_list = []
        logits_steered_list = []
        
        for codice in codici_da_testare:
            logits_base_list.append(estrai_logits_base(codice))
            logits_steered_list.append(estrai_logits_steered(codice))

        mean_base = np.mean(logits_base_list, axis=0) if logits_base_list else []
        mean_steered = np.mean(logits_steered_list, axis=0) if logits_steered_list else []

        num_layers_totali = len(model.model.layers) + 1
        print(f" -> Mappatura Logit Lens su {num_layers_totali} layer completata.")

        for layer_idx in range(num_layers_totali):
            assoluto_base = mean_base[layer_idx] if len(mean_base) > 0 else np.nan
            assoluto_steered = mean_steered[layer_idx] if len(mean_steered) > 0 else np.nan
            
            puro_shift = assoluto_steered - assoluto_base if not pd.isna(assoluto_steered) else np.nan
            
            risultati_sweep_difesa.append({
                "MODELLO": model_name,
                "LAYER": layer_idx,
                "Locus_Iniezione": layer_locus,
                "Base_Steer": assoluto_base,    
                "Delta_Steer": puro_shift       
            })

        print(" Salvataggio dati completato.")

        del model, tokenizer, vettore_tensore
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Errore su {model_name}: {e}")

percorso_csv_finale = "CSV tesi/Logit_Lens/Risultati_Sweep_Steering_FALLIMENTI.csv"
df_risultati = pd.DataFrame(risultati_sweep_difesa)
df_risultati.to_csv(percorso_csv_finale, index=False)

print(f"\n{'='*70}\n ELABORAZIONE FINITA!")
print(f" Il dataset completo è in: {percorso_csv_finale}\n{'='*70}")
