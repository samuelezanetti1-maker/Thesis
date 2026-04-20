from config import models_config
import os
import pandas as pd
import torch
import gc
import numpy as np
import matplotlib.pyplot as plt
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- SETUP E OTTIMIZZAZIONE GPU ---
# Permette a PyTorch di gestire la memoria frammentata sulla GPU, evitando l'errore "Out of Memory"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Indica a HuggingFace dove salvare e andare a pescare i modelli pesanti (evita di riscaricare decine di GB)
os.environ["HF_HOME"] = "/scratch_share/bislab/HF_HUB_CACHE/"

# Creiamo le cartelle fisiche dove salveremo i "referti" della nostra risonanza magnetica
os.makedirs("attivazioni_totali", exist_ok=True)
os.makedirs("grafici_campana", exist_ok=True)

# Dizionario dove salveremo in automatico il layer "bersaglio" (quello col picco L2) per ogni modello
bersagli_steering = {}

# Carichiamo la nostra cartella clinica: il dataset contenente codici SICURI e VULNERABILI
df_modello_TRUE = pd.read_csv("CSV tesi/Dataset/dataset_TRUE.csv")

print("Modelli presenti nel CSV:", df_modello_TRUE['modello'].unique())

# --- FASE 1: CICLO SUI PAZIENTI (MODELLI) ---
for model_name, config in models_config.items():
    try:
        # Filtriamo il CSV per tenere solo gli esempi dove il modello corrente ha risposto correttamente (Baseline pulita)
        df_corretti = df_modello_TRUE[df_modello_TRUE['modello'] == model_name]
        if len(df_corretti) == 0:
            continue

        print(f"-> TROVATI {len(df_corretti)} esempi per {model_name}! Inizio estrazione...")
        print(f"\n{'='*50}\nInizio Analisi Meccanicistica su: {model_name}\n{'='*50}")

        # "Accendiamo" il modello. 
        # Tokenizer: il traduttore che converte le parole in numeri (token)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Model: il "cervello" vero e proprio. Lo carichiamo direttamente nella memoria della scheda video (device_map="auto")
        # in formato a 16 bit (float16) per risparmiare memoria senza perdere precisione geometrica
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=config.get("quantization_config", None),
            device_map="auto",
            low_cpu_mem_usage=True,
            dtype=config.get("dtype", torch.float16)
        )

        # Contiamo quanti layer ha il cervello del modello (es. 32 layer per Llama 8B)
        num_layers = len(model.model.layers)
        print(f"Il modello ha {num_layers} layer. Estrazione in corso...")

        # Creiamo una lista vuota per ogni layer, divisa in base a cosa stava leggendo il modello.
        attivazioni_vulnerabili = {i: [] for i in range(num_layers)}
        attivazioni_sicuri = {i: [] for i in range(num_layers)}

        # Iniziamo a far leggere al modello il nostro dataset riga per riga
        for index, row in df_corretti.iterrows():
            codice = str(row['codice'])
            target = row['target_vero'] # La "verità" presa dal CSV: 'Vulnerabile' o 'Sicuro'

            # Costruiamo il prompt, forzando il modello a prendere una posizione netta (FINAL_VERDICT)
            prompt = f"Analyze this code \n\nCode:\n{codice}, \n start the response EXACTLY with 'FINAL_VERDICT: True' (if vulnerable) or 'FINAL_VERDICT: False' (if 100% secure), followed by a brief summary."
            messages = [
                {"role": "system", "content": "You are a cybersecurity expert. Your task is to find vulnerabilities in the source code."},
                {"role": "user", "content": prompt}
            ]

            # Formattiamo il prompt secondo il dialetto specifico del modello (es. chat template di Llama o Qwen)
            # e lo convertiamo in Tensori (numeri) inviandoli alla GPU (.to(model.device))
            testo_formattato = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([testo_formattato], return_tensors="pt").to(model.device)

            # torch.no_grad() spegne la modalità addestramento (risparmia tantissima memoria)
            with torch.no_grad():
                # LA MACCHINA DELLA RISONANZA MAGNETICA: passiamo l'input al modello.
                # L'argomento chiave è output_hidden_states=True: ordina a PyTorch di non darci solo la risposta finale,
                # ma di salvare e restituirci l'attività elettrica (i tensori) di TUTTI i layer!
                outputs = model(**inputs, output_hidden_states=True)

            # outputs.hidden_states contiene tutto. Il layer 0 è l'embedding delle parole, a noi non interessa.
            # Prendiamo tutti i layer successivi (da 1 a fine)
            hidden_states = outputs.hidden_states[1:] 

            # Entriamo dentro ogni singolo layer per estrarre l'informazione chirurgica
            for layer_idx in range(num_layers):
                # .detach().cpu().numpy() prende il tensore dalla GPU, lo stacca dal grafo computazionale
                # e lo converte in un semplice array Numpy sulla RAM normale.
                # [0, -1, :] significa: prendi la prima frase [0], vai all'ULTIMO token letto [-1], 
                # ed estrai tutti i suoi 4096 numerini (neuroni) [:].
                vettore_ultimo_token = hidden_states[layer_idx][0, -1, :].detach().cpu().numpy()

                # Smistiamo l'array appena estratto nello scatolone giusto in base alla verità del CSV
                if target == 'Vulnerabile':
                    attivazioni_vulnerabili[layer_idx].append(vettore_ultimo_token)
                elif target == 'Sicuro':
                    attivazioni_sicuri[layer_idx].append(vettore_ultimo_token)

        # --- FASE 2: CALCOLO MATEMATICO (ESTRAZIONE DEL CONCETTO) E SALVATAGGIO ---
        print("\nCalcolo centroidi e salvataggio file .npy e .pkl per tutti i layer...")
        magnitudo_layer = []
        
        nome_file_safe = model_name.replace('/', '_')
        
        for layer_idx in range(num_layers):
            # Controllo di sicurezza: se uno scatolone è vuoto, saltiamo per evitare crash matematici
            if len(attivazioni_vulnerabili[layer_idx]) == 0 or len(attivazioni_sicuri[layer_idx]) == 0:
                magnitudo_layer.append(0) 
                continue
                
            # 1. Calcolo dei CENTROIDI: np.mean prende le migliaia di array raccolti e ne calcola la "media esatta" nello spazio 
            media_vuln = np.mean(np.stack(attivazioni_vulnerabili[layer_idx]), axis=0)
            media_sicuro = np.mean(np.stack(attivazioni_sicuri[layer_idx]), axis=0)
            
            # Nasce la freccia della Vulnerabilità (Vettore Steering): la differenza chimica tra i due concetti
            vettore_steering = media_vuln - media_sicuro
            
            # 2. Salviamo questa freccia pura su disco in formato Numpy (.npy)
            # Questo è il file che caricheremo nei prossimi script per fare gli attacchi White-Box (Layer Sweep)
            np.save(f"attivazioni_totali/steering_vector_{nome_file_safe}_layer_{layer_idx}.npy", vettore_steering)
            
            # 3. Creiamo un DataFrame (.pkl) contenente tutte le singole attivazioni raccolte (non solo le medie)
            # Questo file sarà la miniera d'oro per addestrare il Prober (il "sismografo" di sicurezza) offline
            dati_pkl = []
            for vec in attivazioni_vulnerabili[layer_idx]:
                dati_pkl.append({"target_vero": "Vulnerabile", "vettore_attivazione": vec})
            for vec in attivazioni_sicuri[layer_idx]:
                dati_pkl.append({"target_vero": "Sicuro", "vettore_attivazione": vec})
                
            df_pkl = pd.DataFrame(dati_pkl)
            df_pkl.to_pickle(f"attivazioni_totali/vettori_{nome_file_safe}_layer_{layer_idx}.pkl")
            
            # 4. Calcoliamo la NORMA L2: Applichiamo il Teorema di Pitagora (np.linalg.norm) 
            # al Vettore Steering per misurare esattamente "quanto è lunga" la freccia in quel layer
            norma = np.linalg.norm(vettore_steering)
            magnitudo_layer.append(norma)
            
            if layer_idx % 10 == 0:
                print(f"   => Salvato Layer {layer_idx}/{num_layers}...")

        # --- FASE 3: DISEGNO DEL GRAFICO A CAMPANA (LA SINTESI VISIVA) ---
        if magnitudo_layer:
            # Creiamo un grafico cartesiano che mostrerà come la lunghezza della freccia evolve layer dopo layer
            plt.figure(figsize=(10, 6))
            plt.plot(range(num_layers), magnitudo_layer, marker='o', linestyle='-', color='b')
            plt.title(f'Sensibilità alla Vulnerabilità per Layer - {model_name}')
            plt.xlabel('Indice del Layer')
            plt.ylabel('Magnitudo del Vettore Steering (Norma L2)')
            plt.grid(True)
            
            # Salvataggio fisico dell'immagine sul server
            percorso_grafico = f"grafici_campana/{nome_file_safe}_campana.png"
            plt.savefig(percorso_grafico)
            plt.close()
            print(f"Grafico salvato in: {percorso_grafico}")

            # np.argmax cerca automaticamente il layer in cui la L2 è stata più alta (il picco massimo)
            layer_max = np.argmax(magnitudo_layer)
            print(f"IL LAYER PIÙ SENSIBILE È IL: {layer_max} (Magnitudo: {magnitudo_layer[layer_max]:.2f})")
            
            # Salviamo il layer di picco nel nostro dizionario bersagli
            layer_tmp = int(np.argmax(magnitudo_layer).item())
            bersagli_steering[model_name] = layer_tmp

        # --- FASE 4: PULIZIA ESTREMA DELLA SALA OPERATORIA (GPU) ---
        # Essenziale per evitare crash tra l'analisi di un modello e il successivo
        try:
            del model, tokenizer, inputs, outputs
            del attivazioni_vulnerabili, attivazioni_sicuri
        except NameError:
            pass
        gc.collect() # Richiama il Garbage Collector di Python per liberare la RAM
        torch.cuda.empty_cache() # Comando esplicito per Nvidia: svuota tutta la VRAM inutilizzata

    except Exception as e:
        print(f"Errore critico su {model_name}: {e}")
        continue

# Prima di chiudere lo script, salva il dizionario dei layer ottimali in un file JSON
# Così gli script di attacco successivi sapranno automaticamente dove colpire!
with open("bersagli_steering.json", "w") as f:
    json.dump(bersagli_steering, f, indent = 4)