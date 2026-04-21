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

# cattura il tensore mentre passa per il layer scelto e lo copia nella RAM

memoria_layer_centrali = {}

def create_hook(memoria):
    def hook_function(module, input, output):
        if isinstance(output, tuple):
            tensore = output[0]
        else:
            tensore = output
        memoria['corrente'] = tensore.detach().cpu().numpy()  # Copia il tensore nella RAM
    return hook_function

# Hook offensiva (con sottrazione, per la difensiva basta cambiare in +)

def crea_hook_offensiva(vettore_tensore, moltiplicatore):
    # Normalizziamo il vettore per evitare di sovraccaricare troppo il modello e renderlo irriconoscibile, sarà lungo 1
    vettore_norm = vettore_tensore / torch.norm(vettore_tensore)

    def steering_hook_offensiva(module, input, output):
        # 1. Capiamo se l'output è una tupla o un tensore diretto 
        if isinstance(output, tuple):
            tensore_modificato = output[0].clone()
        else:
            tensore_modificato = output.clone()

        vettore_locale = vettore_norm.to(tensore_modificato.device)
        
            
        # 2. Applichiamo lo steering in modo sicuro in base alla forma del tensore
        # Se ha 3 dimensioni [Batch, Sequenza, Neuroni] (Fase di Prefill)
        if len(tensore_modificato.shape) == 3:
            tensore_modificato[:, -1, :] = tensore_modificato[:, -1, :] - (vettore_locale * moltiplicatore)
            
        # Se ha 2 dimensioni [Batch, Neuroni] (Fase di Decodifica con Cache)
        elif len(tensore_modificato.shape) == 2:
            tensore_modificato = tensore_modificato - (vettore_locale * moltiplicatore)
            
        # 3. Restituiamo il risultato nello stesso identico formato in cui è arrivato
        if isinstance(output, tuple):
            return (tensore_modificato,) + output[1:]
        else:
            return tensore_modificato
    return steering_hook_offensiva

### ESTRAZIONE: il modello legge codici sicuri e vulnerabili e fotografiamo lo stato dei neuroni a metà percorso, per capire se c'è una "firma neurale" che distingue i due tipi di codice.


os.makedirs("attivazioni", exist_ok=True)

df_modello_TRUE = pd.read_csv("CSV tesi/Dataset/dataset_TRUE.csv")

for model_name, config in models_config.items():
    try:
        df_corretti = df_modello_TRUE[df_modello_TRUE['modello'] == model_name]
        if len(df_corretti) == 0:
            print(f"\nNessun TP o TN da attaccare per {model_name}")
            continue

        print(f"Trovati {len(df_corretti)} esempi correttamente classificati per {model_name}.")


        # Caricamento modello
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=config.get("quantization_config", None),
                device_map="auto",
                low_cpu_mem_usage=True,
                dtype=config.get("dtype", torch.float16)
            )

        # Calcola automaticamente il layer centrale, indipendentemente dal modello
        layer_to_hook = len(model.model.layers) // 2 
        memoria_layer_centrali[model_name] = layer_to_hook

        print(f"Il modello ha {len(model.model.layers)} layer. Attacchiamo il layer centrale: {layer_to_hook}")

        # Hook per estrarre attivazioni
        memoria_attivazioni = {}

        hook_handle = model.model.layers[layer_to_hook].register_forward_hook(create_hook(memoria_attivazioni))

        vettori_salvati = []

        for index, row in df_corretti.iterrows():
            codice = str(row['codice'])
            target = row['target_vero']

            prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
            messages = [
                {"role": "system", "content": "You are a cybersecurity expert. Your task is to find vulnerabilities in the source code."},
                {"role": "user", "content": prompt}
            ]

            testo_formattato = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([testo_formattato], return_tensors="pt").to(model.device)

            with torch.no_grad():
                model(**inputs)

            vettore_ultimo_token = memoria_attivazioni['corrente'][0, -1, :]  # Prendi l'attivazione dell'ultimo token

            vettori_salvati.append({
                "id_snippet": index,
                "modello": model_name,
                "target_vero": target,
                "vettore_attivazione": vettore_ultimo_token
            })

        #pulizia hook
        hook_handle.remove()

        df_vettori = pd.DataFrame(vettori_salvati)
        nome_file_safe = model_name.replace('/', '_')
        file_output = f"attivazioni/vettori_{nome_file_safe}_layer_{layer_to_hook}.pkl"
        df_vettori.to_pickle(file_output)

        try:
            del model
            del tokenizer
            del inputs
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Errore durante l'elaborazione di {model_name}: {e}")
        continue

    ### CALCOLO DEL VETTORE: estraggo la differenza tra il concetto di "sicuro" e "vulnerabile" [Vulnerability Vector]
# 1- prendo i vettori dei codici vulnerabili e di quelli sicuri
# 2- faccio la media di ciascun gruppo
# 3- sottraggo la media dei due gruppi ottenendo il vector steering

steering_vectors = {}

for model_name in models_config.keys():

    if model_name not in memoria_layer_centrali:
        print(f"Attenzione: non abbiamo attivazioni per {model_name}, saltando il calcolo dello steering vector.")
        continue

    nome_file_safe = model_name.replace('/', '_')
    layer_corretto = memoria_layer_centrali[model_name]
    file_pkl = f"attivazioni/vettori_{nome_file_safe}_layer_{layer_corretto}.pkl"
    if not os.path.exists(file_pkl):
        print(f"File non trovato: {file_pkl} per {model_name}.")
        continue

    # carico i vettori
    df_vettori = pd.read_pickle(file_pkl)
    
    
    lista_vulnerabili = df_vettori[df_vettori['target_vero'] == 'Vulnerabile']['vettore_attivazione'].values
    lista_sicuri = df_vettori[df_vettori['target_vero'] == 'Sicuro']['vettore_attivazione'].values

    # Controllo di sicurezza 
    if len(lista_vulnerabili) == 0 or len(lista_sicuri) == 0:
        print(f" Vettori insufficienti per calcolare lo steering di {model_name} (Vulnerabili: {len(lista_vulnerabili)}, Sicuri: {len(lista_sicuri)}). Salto il calcolo.")
        continue

    vettori_vulnerabili = np.stack(lista_vulnerabili)
    vettori_sicuri = np.stack(lista_sicuri)

    print(f" Trovati {len(vettori_vulnerabili)} vettori vulnerabili e {len(vettori_sicuri)} vettori sicuri per {model_name}.")

    #calcolo centroidi
    media_vulnerabile = np.mean(vettori_vulnerabili, axis=0)
    media_sicuro = np.mean(vettori_sicuri, axis=0)

    #calcolo steering vector
    steering_vector = media_vulnerabile - media_sicuro
    steering_vectors[model_name] = steering_vector

    file_output_npy = f"attivazioni/steering_vector_{nome_file_safe}_layer_{layer_corretto}.npy"
    np.save(file_output_npy, steering_vector)

    print(f"dimensione steering vector per {model_name}: {steering_vector.shape}")

### INIEZIONE (STEERING): inietto il vettore della vulnerabilità in un codice sicuro, per vedere se riesco a farlo classificare come vulnerabile.

lista_moltiplicatori = [3, 5, 8, 10, 15] 
#lista_moltiplicatori = [5]

df_TP = pd.read_csv("CSV tesi/Dataset/dataset_TP.csv")
risultati_attacco_steering = []
risultati_aggregati_centrale = []
log_riassunto_asr = [] # Raccoglitore per il file TXT finale

### CICLO DI ATTACCO
for model_name, config in models_config.items():

    if model_name not in memoria_layer_centrali:
        print(f"Attenzione: non abbiamo attivazioni per {model_name}, saltando l'attacco steering.")
        continue 
    
    try:
        layer_corretto = memoria_layer_centrali[model_name]
        
        print("\n" + "="*60)
        print(f"INIZIO SWEEP STEERING SU: {model_name} | LAYER CENTRALE: {layer_corretto}")
        print("="*60)

        nome_modello_pulito = model_name.replace("/", "_")
        os.makedirs("txt_tesi/Risposte Steering", exist_ok=True)
        percorso_txt_log = f"txt_tesi/Risposte Steering/Log_Risposte_Steering_{nome_modello_pulito}.txt"

        # Inizializziamo il file di log lungo
        with open(percorso_txt_log, "w", encoding="utf-8") as f_log:
            f_log.write(f"=== LOG RISPOSTE STEERING: {model_name} (Layer Centrale: {layer_corretto}) ===\n")
            f_log.write("="*60 + "\n\n")

        # trovo i TP per questo modello
        df_mod = df_TP[df_TP['modello'] == model_name]
        if len(df_mod) == 0:
            print(f"\nNessun TP da attaccare per {model_name}")
            continue

        # carico lo steering vector
        nome_file_safe = model_name.replace('/', '_')
        percorso_vettore = f"attivazioni/steering_vector_{nome_file_safe}_layer_{layer_corretto}.npy"

        if not os.path.exists(percorso_vettore):
            print(f"Vettore di steering non trovato per {model_name}. Percorso cercato: {percorso_vettore}")
            continue

        vettore_numpy = np.load(percorso_vettore)

        # carico modello e tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=config.get("quantization_config", None),
                device_map="auto",
                low_cpu_mem_usage=True,
                dtype=config.get("dtype", torch.float16)
            )

        vettore_tensore = torch.tensor(vettore_numpy, dtype=model.dtype, device=model.device)

        # --- INIZIO CICLO DEI MOLTIPLICATORI ---
        for moltiplicatore in lista_moltiplicatori:
            
            with open(percorso_txt_log, "a", encoding="utf-8") as f_log:
                f_log.write(f"\n>>> TEST CON MOLTIPLICATORE: {moltiplicatore} <<<\n\n")
                
            hook_handle = model.model.layers[layer_corretto].register_forward_hook(
                crea_hook_offensiva(vettore_tensore, moltiplicatore)
            )
            
            attacchi_riusciti = 0
            errori_formattazione = 0

            ### TESTING
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

                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=token_limit, 
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                output_ids = generated_ids[0][len(inputs.input_ids[0]):]
                risposta_steered = tokenizer.decode(output_ids, skip_special_tokens=True)

                risposta_steered = risposta_steered.replace('Ġ', ' ').replace('Ċ', '\n')

                print(f"\n[Snippet ID {index}] Risposta Modello:")
                print(risposta_steered)
                print("-" * 40)
                # -----------------

                # Log della risposta estesa
                with open(percorso_txt_log, "a", encoding="utf-8") as f_log:
                    f_log.write(f"Snippet ID {index}:\n")
                    f_log.write(f"{risposta_steered}\n")
                    f_log.write("-" * 50 + "\n\n")
                
                # Valutazione
                target_predetto_nuovo = evaluate_response(risposta_steered)
                
                if target_predetto_nuovo == "Sicuro":
                    attacchi_riusciti += 1
                elif "FINAL_VERDICT" not in risposta_steered:
                    errori_formattazione += 1
                    
                risultati_attacco_steering.append({
                    "id_snippet": index,
                    "modello": model_name,
                    "layer_attaccato": layer_corretto,
                    "moltiplicatore": moltiplicatore,
                    "target_vero": "Vulnerabile",
                    "predizione_baseline": "Vulnerabile",
                    "predizione_post_steering": target_predetto_nuovo
                })

            # Metriche per questo moltiplicatore
            asr_steering = (attacchi_riusciti / len(df_mod)) * 100
            rateo_errori = (errori_formattazione / len(df_mod)) * 100
            
            stringa_log = f"Moltiplicatore: {moltiplicatore:2d} | ASR: {asr_steering:5.2f}% | Errori (Gibberish): {rateo_errori:5.2f}%"
            log_riassunto_asr.append(f"Modello: {model_name} | Layer Centrale: {layer_corretto} | {stringa_log}")
            
            print(f"   => {stringa_log}")
            risultati_aggregati_centrale.append({
            "modello": model_name,
            "layer": layer_corretto,
            "moltiplicatore": moltiplicatore,
            "asr_percentuale": asr_steering,
            "errori_percentuale": rateo_errori
            })

            # Pulizia post-moltiplicatore
            hook_handle.remove()
            try:
                del inputs
                del generated_ids
            except NameError:
                pass
            torch.cuda.empty_cache()
            
        # --- FINE CICLO MOLTIPLICATORI, PULIZIA MODELLO ---
        try:
            del model
            del tokenizer
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Errore durante l'attacco steering su {model_name}: {e}")
        continue

# Salvataggio CSV Finale
df_steering = pd.DataFrame(risultati_attacco_steering)
df_steering.to_csv("CSV tesi/risultati_attacco_steering.csv", index=False)
print("\nSalvataggio CSV completato in 'CSV tesi/risultati_attacco_steering.csv'")

# Salvataggio TXT Riassuntivo
percorso_txt_riassunto = "CSV tesi/riassunto_ASR_centrale.txt"
with open(percorso_txt_riassunto, "w", encoding="utf-8") as f:
    f.write("=== RISULTATI ABLATION STUDY (ATTACCO LAYER CENTRALE) ===\n")
    f.write("="*60 + "\n\n")
    for riga in log_riassunto_asr:
        f.write(riga + "\n")

# Salvataggio del CSV Aggregato per i grafici
df_aggregato_centrale = pd.DataFrame(risultati_aggregati_centrale)
df_aggregato_centrale.to_csv("CSV tesi/percentuali_aggregate.csv", index=False)
print("CSV aggregato salvato in 'CSV tesi/percentuali_aggregate.csv'")

print(f"Riassunto ASR salvato in: '{percorso_txt_riassunto}'")