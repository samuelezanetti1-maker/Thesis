import os
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import models_config

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/scratch_share/bislab/HF_HUB_CACHE/"

# 1. CARICAMENTO DATI
df_TP = pd.read_csv("CSV tesi/Dataset/dataset_TP.csv")
df_steering_res = pd.read_csv("CSV tesi/Fixed/risultati_attacco_steering.csv")

# Mapping
mapping_dict = df_TP['codice'].to_dict()
df_steering_res['codice_originale'] = df_steering_res['id_snippet'].map(mapping_dict)

layer_locus_per_modello = {
    "Qwen/Qwen2.5-7B-Instruct": 18,
    "Qwen/Qwen2.5-Coder-7B-Instruct": 18,
    "meta-llama/Llama-3.1-8B-Instruct": 15,
    "codellama/CodeLlama-7b-Instruct-hf": 13,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 19,
    "deepseek-ai/deepseek-coder-6.7b-instruct": 16
}

risultati_isomorfismo_sweep = []

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

for model_name, config in models_config.items():
    df_mod = df_steering_res[
        (df_steering_res['modello'] == model_name) & 
        (df_steering_res['predizione_post_steering'] == 'Sicuro')
    ]

    if len(df_mod) == 0:
        continue

    nome_modello_pulito = model_name.replace("/", "_")
    print("\n" + "="*70)
    print(f" SWEEP ISOMORFISMO STEERING (VETTORI MULTIPLI) SU: {model_name} ({len(df_mod)} successi)")
    print("="*70)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
        
        totale_layer = len(model.model.layers)
        num_layers_totali = totale_layer + 1
        layer_locus = layer_locus_per_modello.get(model_name)
        moltiplicatore = df_mod.iloc[0]['moltiplicatore']
        
        # =====================================================================
        # 1. Caricamento del Vettore Locus (Necessario per l'Hook)
        # =====================================================================
        path_vettore_locus = f"attivazioni_totali/steering_vector_{nome_modello_pulito}_layer_{layer_locus}.npy"
        if not os.path.exists(path_vettore_locus):
            print(f" [!] Vettore Locus mancante per {model_name}. Salto.")
            continue
        vettore_locus_np = np.load(path_vettore_locus)
        vettore_tensore_hook = torch.tensor(vettore_locus_np, dtype=model.dtype, device=model.device)

        # =====================================================================
        # 2. Pre-caricamento di TUTTI i vettori per il confronto Isomorfico
        # =====================================================================
        print(" -> Pre-caricamento libreria vettori di steering in VRAM...")
        dizionario_vettori = {}
        vettori_trovati = 0
        for layer_idx in range(num_layers_totali):
            path_steering = f"attivazioni_totali/steering_vector_{nome_modello_pulito}_layer_{layer_idx}.npy"
            if os.path.exists(path_steering):
                vettore_np = np.load(path_steering)
                dizionario_vettori[layer_idx] = torch.tensor(vettore_np, dtype=model.dtype, device=model.device)
                vettori_trovati += 1
            else:
                dizionario_vettori[layer_idx] = None
                
        print(f" -> Trovati {vettori_trovati}/{num_layers_totali} vettori per questo modello.")

        # Funzioni Helper per estrarre tutti i layer
        def get_all_hidden_states(inputs):
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            return torch.stack([layer_state[0, -1, :] for layer_state in out.hidden_states])

        def get_all_hidden_states_steered(inputs, locus, hook_func):
            handle = model.model.layers[locus].register_forward_hook(hook_func)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            handle.remove()
            return torch.stack([layer_state[0, -1, :] for layer_state in out.hidden_states])

        accumulatore_sim = {i: [] for i in range(num_layers_totali)}

        for index, row in df_mod.iterrows():
            codice_pulito = str(row['codice_originale']) 
            
            prompt = f"Analyze this code \n\nCode:\n{codice_pulito}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
            messages = [{"role": "system", "content": "You are a cybersecurity expert."}, {"role": "user", "content": prompt}]
            testo_formattato = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([testo_formattato], return_tensors="pt").to(model.device)
            
            # PASSAGGIO A: Forward Pass Base (Tutti i layer)
            h_clean_all = get_all_hidden_states(inputs)

            # PASSAGGIO B: Forward Pass Steered (Tutti i layer)
            hook = crea_hook_offensiva(vettore_tensore_hook, moltiplicatore)
            h_steered_all = get_all_hidden_states_steered(inputs, layer_locus, hook)

            # PASSAGGIO C: Calcolo Shift Globale dello Steering
            shift_all = h_steered_all - h_clean_all

            # PASSAGGIO D: Calcolo Cosine Similarity Layer per Layer
            for layer_idx in range(num_layers_totali):
                vettore_riferimento_locale = dizionario_vettori[layer_idx]
                
                if vettore_riferimento_locale is not None:
                    shift_layer = shift_all[layer_idx]
                    cos_sim = F.cosine_similarity(shift_layer, vettore_riferimento_locale, dim=0).item()
                    accumulatore_sim[layer_idx].append(cos_sim)

        # Calcolo delle Medie Finali
        for layer_idx in range(num_layers_totali):
            valori_layer = accumulatore_sim[layer_idx]
            if len(valori_layer) > 0:
                media_sim = np.mean(valori_layer)
                media_assoluta = np.mean([abs(x) for x in valori_layer])
                
                risultati_isomorfismo_sweep.append({
                    "modello": model_name,
                    "layer_analizzato": layer_idx,
                    "layer_iniezione_locus": layer_locus, # Manteniamo il tracking di dove abbiamo iniettato
                    "media_cosine_similarity": media_sim,
                    "media_valore_assoluto": media_assoluta
                })

        del model, tokenizer
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Errore su {model_name}: {e}")

os.makedirs("CSV tesi/Isomorfismo_Sweep", exist_ok=True)
df_finale = pd.DataFrame(risultati_isomorfismo_sweep)
df_finale.to_csv("CSV tesi/Isomorfismo_Sweep/sweep_isomorfismo_multi_vector_Steering.csv", index=False)
print("\n[+] Salvataggio completato in 'CSV tesi/Isomorfismo_Sweep/sweep_isomorfismo_multi_vector_Steering.csv'")