from config import models_config
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/scratch_share/bislab/HF_HUB_CACHE/"

import pandas as pd
import torch
import gc
import random
import re
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, LogitsProcessor
from utils import evaluate_response_logit

class BiasLogitsProcessor(LogitsProcessor):
    def __init__(self, token_ids_boost, token_ids_penalize, boost_val, penalty_val):
        self.token_ids_boost = token_ids_boost
        self.token_ids_penalize = token_ids_penalize
        # Mi assicuro che siano float
        self.boost_val = float(boost_val)
        self.penalty_val = float(penalty_val)

    def __call__(self, input_ids, scores):
        for token_id in self.token_ids_boost:
            scores[:, token_id] += self.boost_val
        for token_id in self.token_ids_penalize:
            scores[:, token_id] -= self.penalty_val
        return scores
        
lb_TP = pd.read_csv("CSV tesi/Dataset/dataset_TP.csv")
risultati_attacco_logits = []

# Valori da testare
valori_bias = [4, 6, 8, 10, 15]

for model_name, config in models_config.items():
    print(f"\n" + "="*50)
    print(f"Attacco logits su: {model_name}")

    # PRENDO I 20 ESEMPI *PER QUESTO SPECIFICO MODELLO*
    df_mod = lb_TP[lb_TP['modello'] == model_name].head(20)
    
    if len(df_mod) == 0:
        print(f"Nessun TP da attaccare per {model_name}")
        continue

    nome_modello_pulito = model_name.replace("/", "_")

    # Caricamento modello e tokenizer (eseguito UNA volta per modello)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=config.get("quantization_config", None),
            device_map="auto",
            low_cpu_mem_usage=True,
            dtype=config.get("dtype", torch.float16)
        )

    # Identifico i token ID
    parole_sicuro = ["False", "false", "FALSE", " False", " false", " FALSE", "Sicuro", "Secure"]
    parole_vuln =  ["True", "true", "TRUE", " True", " true", " TRUE", "Vulnerabile", "Vulnerable"]

    token_boost = list(set([t for p in parole_sicuro for t in tokenizer.encode(p, add_special_tokens=False)]))
    token_penalize = list(set([t for p in parole_vuln for t in tokenizer.encode(p, add_special_tokens=False)]))

    # CICLO SUI DIVERSI VALORI DI BIAS
    for bias in valori_bias:
        print(f"\n---> Testando Bias: {bias} su {model_name}")
        
        # File di log differenziato per bias
        percorso_txt = f"logit_test/Log_{nome_modello_pulito}_bias_{bias}.txt"
        with open(percorso_txt, "w", encoding="utf-8") as f_log:
            f_log.write(f"=== LOG RISPOSTE LOGIT BIAS: {model_name} | BIAS: {bias} ===\n")
            f_log.write("="*60 + "\n\n")

        # Iniezione ai logits usando la variabile 'bias'
        hacker_logits_processor = BiasLogitsProcessor(
            token_ids_boost=token_boost,
            token_ids_penalize=token_penalize,
            boost_val=bias,    
            penalty_val=bias   
        )

        logits_processor_list = LogitsProcessorList([hacker_logits_processor])
        attacchi_riusciti = 0
        attacchi_non_validi = 0

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
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=token_limit, 
                    do_sample=False,
                    logits_processor=logits_processor_list,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            output_ids = generated_ids[0][len(inputs.input_ids[0]):]
            risposta_steered = tokenizer.decode(output_ids, skip_special_tokens=True)

            with open(percorso_txt, "a", encoding="utf-8") as f_log:
                f_log.write(f"Snippet ID {index}:\n{risposta_steered}\n")
                f_log.write("-" * 50 + "\n\n")
            
            target_predetto_nuovo = evaluate_response_logit(risposta_steered)
            
            if target_predetto_nuovo == "Sicuro":
                attacchi_riusciti += 1
            if target_predetto_nuovo == "Non Classificato":
                attacchi_non_validi += 1
                
            risultati_attacco_logits.append({
                "id_snippet": index,
                "modello": model_name,
                "valore_bias": bias, # IMPORTANTE: traccia il valore nel CSV
                "target_vero": "Vulnerabile",
                "predizione_baseline": "Vulnerabile",
                "predizione_post_bias": target_predetto_nuovo
            })
            
            # Pulisco solo le variabili temporanee del prompt
            del inputs
            del generated_ids
            torch.cuda.empty_cache()
            
        asr_logits = (attacchi_riusciti / len(df_mod)) * 100
        print(f"ASR su {model_name} (Bias {bias}): {asr_logits:.2f}% ({attacchi_riusciti}/{len(df_mod)})")

    # Pulizia completa del modello SOLO quando passo al modello successivo
    try:
        del model
        del tokenizer
    except NameError:
        pass
    gc.collect()
    torch.cuda.empty_cache()

# Salvataggio unico per tutti i modelli e tutti i bias
df_logits = pd.DataFrame(risultati_attacco_logits)
df_logits.to_csv("logit_test/risultati_attacco_logits_ablation.csv", index=False)
print("\nScansione completata. Risultati salvati.")