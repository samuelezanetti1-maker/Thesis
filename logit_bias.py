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

### Voglio andare a modificare i token con parola True dando penalità e invece ai False do un boost, cerco di forzare così la risposta.

class BiasLogitsProcessor(LogitsProcessor):
    def __init__(self, token_ids_boost, token_ids_penalize, boost_val, penalty_val):
        self.token_ids_boost = token_ids_boost
        self.token_ids_penalize = token_ids_penalize
        self.boost_val = boost_val
        self.penalty_val = penalty_val

    def __call__(self, input_ids, scores):
        for token_id in self.token_ids_boost:
            scores[:, token_id] += self.boost_val
        for token_id in self.token_ids_penalize:
            scores[:, token_id] -= self.penalty_val
        return scores
        
lb_TP = pd.read_csv("CSV tesi/Dataset/dataset_TP.csv")
risultati_attacco_logits = []

for model_name, config in models_config.items():
    print(f"attacco logits su: {model_name}")

    df_mod = lb_TP[lb_TP['modello'] == model_name]
    if len(df_mod) == 0:
        print(f"Nessun TP da attaccare per {model_name}")
        continue

    nome_modello_pulito = model_name.replace("/", "_")
    percorso_txt = f"logit_test/Log_Risposte_Logits_{nome_modello_pulito}_4.txt"

    with open(percorso_txt, "w", encoding="utf-8") as f_log:
        f_log.write(f"=== LOG RISPOSTE LOGIT BIAS: {model_name} ===\n")
        f_log.write("="*60 + "\n\n")

    #caricamento modello e tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=config.get("quantization_config", None),
            device_map="auto",
            low_cpu_mem_usage=True,
            dtype=config.get("dtype", torch.float16)
        )

# Identifico i token ID per "True" e "False"
    parole_sicuro = ["False", "false", "FALSE", " False", " false", " FALSE", "Sicuro", "Secure"]
    parole_vuln =  ["True", "true", "TRUE", " True", " true", " TRUE", "Vulnerabile", "Vulnerable"]

    token_boost = []
    token_penalize = []

    for p in parole_sicuro:
        token_boost.extend(tokenizer.encode(p, add_special_tokens=False))

    for p in parole_vuln:
        token_penalize.extend(tokenizer.encode(p, add_special_tokens=False))

    ##Rimuovo duplicati
    token_boost = list(set(token_boost))
    token_penalize = list(set(token_penalize))

    #iniezione ai logits (boost e penalty indicano la forza)
    hacker_logits_processor = BiasLogitsProcessor(
        token_ids_boost=token_boost,
        token_ids_penalize=token_penalize,
        boost_val=6.0,    # aumenta la probabilità di "False"
        penalty_val=6.0   # diminuisce la probabilità di "True"
    )

    logits_processor_list = LogitsProcessorList([hacker_logits_processor])
    attacchi_riusciti = 0
    attacchi_non_validi = 0
    print(f" trovati {len(df_mod)} esempi da attaccare con manipolazione dei logits per {model_name}.")

    ##generazione forzata
    for index, row in df_mod.iterrows():
        print(f"Analizzando lo snippet ID {index}...")
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

        ## salvataggio risposta in txt
        with open(percorso_txt, "a", encoding="utf-8") as f_log:
            f_log.write(f"Snippet ID {index}:\n")
            f_log.write(f"{risposta_steered}\n")
            f_log.write("-" * 50 + "\n\n")
        
        risposta_pulita = risposta_steered.replace('\n', ' ')
        print(f"Risposta: {risposta_pulita[:80]}...")
        
        target_predetto_nuovo = evaluate_response_logit(risposta_steered)
        
        # L'attacco ha successo se dice "Sicuro" (False)
        if target_predetto_nuovo == "Sicuro":
            attacchi_riusciti += 1
        if target_predetto_nuovo == "Non Classificato":
            attacchi_non_validi += 1
            
        risultati_attacco_logits.append({
            "id_snippet": index,
            "modello": model_name,
            "target_vero": "Vulnerabile",
            "predizione_baseline": "Vulnerabile",
            "predizione_post_bias": target_predetto_nuovo
        })
        
    asr_logits = (attacchi_riusciti / len(df_mod)) * 100
    print(f"\nASR (Attacco Logit Bias Riuscito) su {model_name}: {asr_logits:.2f}% ({attacchi_riusciti}/{len(df_mod)})")
    print(f"\nAttacchi non validi {attacchi_non_validi} su {len(df_mod)}\n")
    # Pulizia VRAM
    try:
        del model
        del tokenizer
        del inputs
        del generated_ids
    except NameError:
        pass
    gc.collect()
    torch.cuda.empty_cache()

# Salvataggio
df_logits = pd.DataFrame(risultati_attacco_logits)
df_logits.to_csv(f"logit_test/risultati_attacco_logits_4.csv", index=False)

