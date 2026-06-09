from pyexpat.errors import messages

from config import models_config
import os
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/scratch_share/bislab/HF_HUB_CACHE/"

file_attacchi = "CSV tesi/Fixed/risultati_attacco_advanced.csv" 
df_attacchi = pd.read_csv(file_attacchi)

# Filtriamo solo gli attacchi che hanno avuto SUCCESSO (Falsi Negativi)
df_successi = df_attacchi[df_attacchi['target_predetto_adv'] == 'Sicuro']

risultati_isomorfismo = []

for model_name, config in models_config.items():
    
    df_mod = df_successi[df_successi['modello'] == model_name]
    if len(df_mod) == 0:
        continue

    nome_modello_pulito = model_name.replace("/", "_")
    print("\n" + "="*70)
    print(f" RICERCA TEORIA UNIFICATA SU: {model_name} ({len(df_mod)} attacchi riusciti)")
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

        layer_ottimali = {
            "Qwen/Qwen2.5-7B-Instruct": 18, 
            "Qwen/Qwen2.5-Coder-7B-Instruct": 18,
            "meta-llama/Llama-3.1-8B-Instruct": 15,
            "codellama/CodeLlama-7b-Instruct-hf": 13,
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 19,
            "deepseek-ai/deepseek-coder-6.7b-instruct": 16,
            
        }

        layer_locus = layer_ottimali.get(model_name, int(len(model.model.layers) * 0.55))
        
        print(f" -> Layer chirurgico selezionato per l'analisi: {layer_locus}")

        # Carico il Vettore di Steering Matematico
        percorso_vettore = f"attivazioni_totali/steering_vector_{nome_modello_pulito}_layer_{layer_locus}.npy"
        if not os.path.exists(percorso_vettore):
            continue
            
        vettore_steering_originale = torch.tensor(np.load(percorso_vettore), dtype=model.dtype).to(model.device)

        cosine_similarities = []

        for index, row in df_mod.iterrows():
            codice_pulito = str(row['codice_originale']) 
            codice_hackerato = str(row['codice_perturbato'])

            def get_hidden_state(codice):
                prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
                messages = [
                    {"role": "system", "content": "You are a cybersecurity expert. Your task is to find vulnerabilities in the source code."},
                    {"role": "user", "content": prompt}
                ]

                testo_formattato = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([testo_formattato], return_tensors="pt").to(model.device)
            
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                
                # Estraiamo l'attivazione dell'ultimo token al layer bersaglio
                hidden_state = outputs.hidden_states[layer_locus][0, -1, :]
                return hidden_state

            # 1. Fotografa il pensiero pulito
            h_clean = get_hidden_state(codice_pulito)
            
            # 2. Fotografa il pensiero hackerato testualmente
            h_adv = get_hidden_state(codice_hackerato)

            # 3. Calcola lo Shift (La direzione in cui il testo ha spinto il modello)
            shift_text = h_adv - h_clean

            # 4. Calcola la Cosine Similarity
            # Usiamo dim=0 perché stiamo confrontando due vettori 1D
            cos_sim = F.cosine_similarity(shift_text, vettore_steering_originale, dim=0).item()
            
            cosine_similarities.append(cos_sim)

            risultati_isomorfismo.append({
                "id_snippet": index,
                "modello": model_name,
                "layer_analizzato": layer_locus,
                "cosine_similarity": cos_sim,
                "valore_assoluto_allineamento": abs(cos_sim)
            })

        media_sim = np.mean(cosine_similarities)
        media_assoluta = np.mean([abs(x) for x in cosine_similarities])
        
        print(f" -> Cosine Similarity Media (Direzionale): {media_sim:.4f}")
        print(f" -> Allineamento sull'asse (Valore Assoluto): {media_assoluta:.4f}")
        
        # Nota: In spazi a 4096 dimensioni, due vettori casuali hanno una cosine similarity quasi a 0.00
        # Qualsiasi valore oltre 0.05 o sotto -0.05 è un forte segnale statistico

        del model, tokenizer
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Errore su {model_name}: {e}")
        continue

df_finale = pd.DataFrame(risultati_isomorfismo)
df_finale.to_csv("CSV tesi/best_convergenza_isomorfismo_advanced.csv", index=False)
print("\n[+] Dati sull'isomorfismo salvati in 'CSV tesi/best_convergenza_isomorfismo_advanced.csv'")
