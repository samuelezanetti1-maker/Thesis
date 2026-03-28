import os
import pandas as pd
import torch
import gc
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import evaluate_response
from config import models_config

try:
    with open("bersagli_steering.json", "r") as f:
        bersagli_steering = json.load(f)
    print(f"Bersagli caricati con successo: {bersagli_steering}")
except FileNotFoundError:
    print("[!] Errore: Il file 'bersagli_steering.json' non esiste. Esegui prima analisi_mec.py!")
    exit()

#lista_moltiplicatori = [1, 3, 5, 8, 10, 15, 20]
lista_moltiplicatori = [5] 

df_TP = pd.read_csv("CSV tesi/Dataset/dataset_TP.csv")
risultati_attacco_mirato = []
log_riassunto_asr = []
risultati_aggregati_mirato = []

# Hook offensiva 
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


print("\n" + "="*60)
print("INIZIO ABLATION STUDY")
print("="*60)

for model_name, config in models_config.items():
    if model_name not in bersagli_steering:
        continue 
        
    layer_mirato = bersagli_steering[model_name]
    
    df_mod = df_TP[df_TP['modello'] == model_name]
    if len(df_mod) == 0:
        continue

    print(f"\n-> Attacco {model_name} sul suo LAYER PIÙ SENSIBILE: {layer_mirato}")

    # CARICHIAMO IL VETTORE GIÀ CALCOLATO DA analisi_mec.py
    nome_file_safe = model_name.replace('/', '_')
    percorso_vettore = f"attivazioni_totali/steering_vector_{nome_file_safe}_layer_{layer_mirato}.npy"

    if not os.path.exists(percorso_vettore):
        print(f"[!] Vettore non trovato: {percorso_vettore}")
        continue

    vettore_numpy = np.load(percorso_vettore)

    # Caricamento Modello
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=config.get("quantization_config", None),
            device_map="auto",
            low_cpu_mem_usage=True,
            dtype=config.get("dtype", torch.float16)
        )

    vettore_tensore = torch.tensor(vettore_numpy, dtype=model.dtype, device=model.device)

    # APPLICHIAMO L'HOOK AL LAYER SPECIFICO TROVATO DALLA DIAGNOSI

    nome_modello_pulito = model_name.replace("/", "_")
    os.makedirs("txt_tesi/Risposte Snipe", exist_ok=True)
    percorso_txt_log = f"txt_tesi/Risposte Snipe/Log_Risposte_Snipe_{nome_modello_pulito}.txt"

    with open(percorso_txt_log, "w", encoding="utf-8") as f_log:
        f_log.write(f"=== LOG RISPOSTE SNIPE STEERING: {model_name} (Layer Mirato: {layer_mirato}) ===\n")
        f_log.write("="*60 + "\n\n")

    for moltiplicatore in lista_moltiplicatori:

        with open(percorso_txt_log, "a", encoding="utf-8") as f_log:
            f_log.write(f"\n>>> TEST CON MOLTIPLICATORE: {moltiplicatore} <<<\n\n")

        hook_handle = model.model.layers[layer_mirato].register_forward_hook(
            crea_hook_offensiva(vettore_tensore, moltiplicatore)
        )
        
        attacchi_riusciti = 0
        errori_formattazione = 0

        # TESTING SUI TP
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

            generated_ids = model.generate(**inputs, max_new_tokens=token_limit, do_sample=False)
            output_ids = generated_ids[0][len(inputs.input_ids[0]):]
            risposta_steered = tokenizer.decode(output_ids, skip_special_tokens=True)

            risposta_steered = risposta_steered.replace('Ġ', ' ').replace('Ċ', '\n')
            
            print(f"\n[Snippet ID {index}] Risposta Modello:")
            print(risposta_steered)
            print("-" * 40)

            with open(percorso_txt_log, "a", encoding="utf-8") as f_log:
                f_log.write(f"Snippet ID {index}:\n")
                f_log.write(f"{risposta_steered}\n")
                f_log.write("-" * 50 + "\n\n")
            
            target_predetto_nuovo = evaluate_response(risposta_steered)
            
            # L'attacco ha successo se ora dice "Sicuro" (False)
            if target_predetto_nuovo == "Sicuro":
                attacchi_riusciti += 1
            elif "FINAL_VERDICT" not in risposta_steered:
                errori_formattazione += 1
                
            risultati_attacco_mirato.append({
                "id_snippet": index,
                "modello": model_name,
                "layer_attaccato": layer_mirato,
                "moltiplicatore": moltiplicatore,
                "target_vero": "Vulnerabile",
                "predizione_baseline": "Vulnerabile",
                "predizione_post_steering": target_predetto_nuovo
            })

        asr_mirato = (attacchi_riusciti / len(df_mod)) * 100
        rateo_errori = (errori_formattazione / len(df_mod)) * 100

        stringa_log = f"Moltiplicatore: {moltiplicatore:2d} | ASR: {asr_mirato:5.2f}% | Errori (Gibberish): {rateo_errori:5.2f}%"
        log_riassunto_asr.append(f"Modello: {model_name} | Layer Mirato: {layer_mirato} | {stringa_log}")
        print(f"   => {stringa_log}")

        risultati_aggregati_mirato.append({
            "modello": model_name,
            "layer": layer_mirato,
            "moltiplicatore": moltiplicatore,
            "asr_percentuale": asr_mirato,
            "errori_percentuale": rateo_errori
        })

        # Pulizia
        hook_handle.remove()
        del inputs, generated_ids
        torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# Salvataggio
df_mirato = pd.DataFrame(risultati_attacco_mirato)
df_mirato.to_csv("CSV tesi/risultati_attacco_mirato_steering.csv", index=False)

df_aggregato = pd.DataFrame(risultati_aggregati_mirato)
df_aggregato.to_csv("CSV tesi/percentuali_aggregate_mirato.csv", index=False)

percorso_txt_riassunto = "CSV tesi/riassunto_ASR_mirato.txt"
with open(percorso_txt_riassunto, "w", encoding="utf-8") as f:
    f.write("=== RISULTATI ABLATION STUDY (ATTACCO MIRATO) ===\n")
    f.write("="*60 + "\n\n")
    for riga in log_riassunto_asr:
        f.write(riga + "\n")