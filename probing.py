from config import models_config
import os
import pandas as pd
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- CONFIGURAZIONE ---
# Inserisci il nome pulito del modello esattamente come compare nei file .pkl
# Esempio: "Qwen_Qwen2.5-7B-Instruct" o "meta-llama_Llama-3.1-8B-Instruct"
for model_name, config in models_config.items():

    nome_modello = model_name
    cartella_attivazioni = "attivazioni_totali" # La cartella che hai scaricato con scp

    # Cerchiamo tutti i file .pkl per questo modello
    pattern_ricerca = "attivazioni_totali/*" + nome_modello.replace("/", "_") + "_layer_*.pkl"
    file_trovati = glob.glob(pattern_ricerca)

    if not file_trovati:
        print(f"Nessun file trovato per {nome_modello} in {cartella_attivazioni}/")
        exit()

    risultati_probing = []

    print(f"Inizio Probing su {nome_modello} ({len(file_trovati)} layer trovati)...\n")

    # --- CICLO SUI LAYER ---
    for file_pkl in file_trovati:
        # Estraiamo il numero del layer dal nome del file
        match = re.search(r'layer_(\d+)\.pkl', file_pkl)
        if not match:
            continue
        layer_idx = int(match.group(1))
        
        # Caricamento dati
        df = pd.read_pickle(file_pkl)
        
        # Rimuoviamo eventuali righe corrotte
        df = df.dropna(subset=['vettore_attivazione', 'target_vero'])
        
        if len(df) < 10:
            continue # Troppi pochi dati per fare machine learning
            
        # Estraiamo X (le feature: l'array Numpy delle attivazioni) e y (le etichette)
        X = np.stack(df['vettore_attivazione'].values)
        y = df['target_vero'].values
        
        # Dividiamo in train (80%) e test (20%) in modo stratificato
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Addestriamo il Prober (un semplice classificatore lineare)
        # Aumentiamo max_iter perché i vettori a volte sono complessi da separare
        prober = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        prober.fit(X_train, y_train)
        
        # Valutiamo l'accuratezza sul Test Set
        predizioni = prober.predict(X_test)
        accuratezza = accuracy_score(y_test, predizioni) * 100
        
        risultati_probing.append((layer_idx, accuratezza))
        print(f"Layer {layer_idx:2d} -> Accuratezza Probing: {accuratezza:.2f}%")

    # Ordinamo i risultati in base al numero del layer
    risultati_probing.sort(key=lambda x: x[0])
    layers = [x[0] for x in risultati_probing]
    accuracies = [x[1] for x in risultati_probing]

    # --- PLOTTING PER LA TESI ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Disegniamo la linea
    plt.plot(layers, accuracies, marker='o', linestyle='-', color='b', linewidth=2, markersize=6)

    # Linea di Baseline (Caso Random) a 50%
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='Random Baseline (50%)')

    plt.title(f'Probing Accuracy per Layer\nModello: {nome_modello.replace("_", "/")}', fontsize=14, fontweight='bold')
    plt.xlabel('Indice del Layer', fontsize=12)
    plt.ylabel('Accuratezza (%) nella separazione Sicuro/Vulnerabile', fontsize=12)
    plt.ylim(30, 105) # Fissiamo l'asse Y per una lettura chiara
    plt.xticks(layers) # Mostriamo tutti i layer sull'asse X
    plt.legend()
    plt.tight_layout()

    # Salviamo il grafico in alta qualità
    nome_file_grafico = f"Probing_{nome_modello.replace('/', '_')}.png"
    plt.savefig(nome_file_grafico, dpi=300)
    print(f"\nGrafico salvato con successo: {nome_file_grafico}")

    