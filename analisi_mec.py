from config import models_config
import os
import pandas as pd
import torch
import gc
import numpy as np
import matplotlib.pyplot as plt
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/scratch_share/bislab/HF_HUB_CACHE/"

# cartelle per i risultati
os.makedirs("attivazioni_totali", exist_ok=True)
os.makedirs("grafici_campana", exist_ok=True)

bersagli_steering = {}

df_modello_TRUE = pd.read_csv("CSV tesi/Dataset/dataset_TRUE.csv")

print("Modelli presenti nel CSV:", df_modello_TRUE['modello'].unique())

for model_name, config in models_config.items():
    try:
        df_corretti = df_modello_TRUE[df_modello_TRUE['modello'] == model_name]
        if len(df_corretti) == 0:
            continue

        print(f"-> TROVATI {len(df_corretti)} esempi per {model_name}! Inizio estrazione...")

        print(f"\n{'='*50}\nInizio Analisi Meccanicistica su: {model_name}\n{'='*50}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=config.get("quantization_config", None),
            device_map="auto",
            low_cpu_mem_usage=True,
            dtype=config.get("dtype", torch.float16)
        )

        num_layers = len(model.model.layers)
        print(f"Il modello ha {num_layers} layer. Estrazione in corso...")

        # Dizionari per accumulare le attivazioni per ogni layer
        # Struttura: { numero_layer: [lista_vettori_ultimo_token] }
        attivazioni_vulnerabili = {i: [] for i in range(num_layers)}
        attivazioni_sicuri = {i: [] for i in range(num_layers)}

        for index, row in df_corretti.iterrows():
            codice = str(row['codice'])
            target = row['target_vero'] # Può essere 'Vulnerabile' o 'Sicuro'

            prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
            messages = [
                {"role": "system", "content": "You are a cybersecurity expert. Your task is to find vulnerabilities in the source code."},
                {"role": "user", "content": prompt}
            ]

            testo_formattato = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([testo_formattato], return_tensors="pt").to(model.device)

            with torch.no_grad():
                # LA MAGIA È QUI: output_hidden_states=True
                outputs = model(**inputs, output_hidden_states=True)

            # outputs.hidden_states è una tupla che contiene i tensori di tutti i layer
            # Nota: il layer 0 spesso è l'embedding iniziale
            hidden_states = outputs.hidden_states[1:] # Salto l'embedding iniziale

            for layer_idx in range(num_layers):
                # Estraiamo l'attivazione dell'ultimo token per questo specifico layer
                # shape: [batch, sequenza, neuroni] -> prendiamo [0, -1, :]
                vettore_ultimo_token = hidden_states[layer_idx][0, -1, :].detach().cpu().numpy()

                if target == 'Vulnerabile':
                    attivazioni_vulnerabili[layer_idx].append(vettore_ultimo_token)
                elif target == 'Sicuro':
                    attivazioni_sicuri[layer_idx].append(vettore_ultimo_token)

        # --- FASE 2: Calcolo e Salvataggio MASSIVO (.npy e .pkl) ---
        print("\nCalcolo centroidi e salvataggio file .npy e .pkl per tutti i layer...")
        magnitudo_layer = []
        
        nome_file_safe = model_name.replace('/', '_')
        
        for layer_idx in range(num_layers):
            if len(attivazioni_vulnerabili[layer_idx]) == 0 or len(attivazioni_sicuri[layer_idx]) == 0:
                magnitudo_layer.append(0) # Per non sfalsare il grafico
                continue
                
            # 1. Centroidi e Vettore Steering
            media_vuln = np.mean(np.stack(attivazioni_vulnerabili[layer_idx]), axis=0)
            media_sicuro = np.mean(np.stack(attivazioni_sicuri[layer_idx]), axis=0)
            vettore_steering = media_vuln - media_sicuro
            
            # 2. Salvataggio Vettore Steering (.npy) per il Layer Sweep sul cluster
            np.save(f"attivazioni_totali/steering_vector_{nome_file_safe}_layer_{layer_idx}.npy", vettore_steering)
            
            # 3. Creazione e Salvataggio del DataFrame (.pkl) per il Probing in locale
            dati_pkl = []
            for vec in attivazioni_vulnerabili[layer_idx]:
                dati_pkl.append({"target_vero": "Vulnerabile", "vettore_attivazione": vec})
            for vec in attivazioni_sicuri[layer_idx]:
                dati_pkl.append({"target_vero": "Sicuro", "vettore_attivazione": vec})
                
            df_pkl = pd.DataFrame(dati_pkl)
            df_pkl.to_pickle(f"attivazioni_totali/vettori_{nome_file_safe}_layer_{layer_idx}.pkl")
            
            # 4. Calcolo Norma L2 per la campana
            norma = np.linalg.norm(vettore_steering)
            magnitudo_layer.append(norma)
            
            if layer_idx % 10 == 0:
                print(f"   => Salvato Layer {layer_idx}/{num_layers}...")

        # --- FASE 3: Disegno del Grafico a Campana ---
        if magnitudo_layer:
            plt.figure(figsize=(10, 6))
            plt.plot(range(num_layers), magnitudo_layer, marker='o', linestyle='-', color='b')
            plt.title(f'Sensibilità alla Vulnerabilità per Layer - {model_name}')
            plt.xlabel('Indice del Layer')
            plt.ylabel('Magnitudo del Vettore Steering (Norma L2)')
            plt.grid(True)
            
            # Salvataggio del grafico
            percorso_grafico = f"grafici_campana/{nome_file_safe}_campana.png"
            plt.savefig(percorso_grafico)
            plt.close()
            print(f"Grafico salvato in: {percorso_grafico}")

            # Trova il layer più sensibile (il picco della campana)
            layer_max = np.argmax(magnitudo_layer)
            print(f"IL LAYER PIÙ SENSIBILE È IL: {layer_max} (Magnitudo: {magnitudo_layer[layer_max]:.2f})")
            
            layer_tmp = int(np.argmax(magnitudo_layer).item())
            bersagli_steering[model_name] = layer_tmp

        # Pulizia memoria
        try:
            del model, tokenizer, inputs, outputs
            del attivazioni_vulnerabili, attivazioni_sicuri
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Errore critico su {model_name}: {e}")
        continue

with open("bersagli_steering.json", "w") as f:
    json.dump(bersagli_steering, f, indent = 4)

