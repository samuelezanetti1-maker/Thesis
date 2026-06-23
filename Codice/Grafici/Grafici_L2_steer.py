import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import models_config
import pandas as pd

#### BLU SICURO
#### ROSSO VULNERABILE


sns.set_theme(style="whitegrid", context="talk")

def calcola_coordinate_e_angoli(a, b, c):
    #if a + b <= c or a + c <= b or b + c <= a:
     #   raise ValueError(f"I lati {a}, {b}, {c} non formano un triangolo valido!")

    x_A, y_A = 0, 0
    x_B, y_B = c, 0
    x_C = (c**2 + b**2 - a**2) / (2 * c)
    y_C = np.sqrt(abs(b**2 - x_C**2)) 

    angolo_A = np.degrees(np.arccos((b**2 + c**2 - a**2) / (2 * b * c)))
    angolo_B = np.degrees(np.arccos((a**2 + c**2 - b**2) / (2 * a * c)))
    angolo_C = 180.0 - angolo_A - angolo_B 

    x_coords = [x_A, x_B, x_C, x_A]
    y_coords = [y_A, y_B, y_C, y_A]

    return (x_coords, y_coords), (angolo_A, angolo_B, angolo_C)


def compara_due_triangoli(lati_t1, lati_t2, nome, nome_t1="Triangolo 1", nome_t2="Triangolo 2"):
    coords_1, angoli_1 = calcola_coordinate_e_angoli(*lati_t1)
    coords_2, angoli_2 = calcola_coordinate_e_angoli(*lati_t2)

    plt.figure(figsize=(10, 7))

    # Uso i colori delle palette predefinite di seaborn (più accademici)
    palette = sns.color_palette("deep")
    colore_1 = palette[0] # Solitamente un blu elegante
    colore_2 = palette[3] # Solitamente un rosso/mattone elegante

    # Disegno Triangolo 1
    plt.plot(coords_1[0], coords_1[1], marker='o', color=colore_1, label=nome_t1)
    plt.fill(coords_1[0], coords_1[1], color=colore_1, alpha=0.2)

    # Disegno Triangolo 2
    plt.plot(coords_2[0], coords_2[1], marker='s', color=colore_2, linestyle='--', label=nome_t2)
    plt.fill(coords_2[0], coords_2[1], color=colore_2, alpha=0.2)

    plt.axis('equal') 
    
        
    plt.title(f"{nome}\n Distanze L2", fontweight='bold', pad=20)
    plt.xlabel("L2 Proiettata (Asse X)", labelpad=15)
    plt.ylabel("L2 Proiettata (Asse Y)", labelpad=15)

    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()


df = pd.read_csv("Distanze_L2_values.csv")

layer_per_modello = {
            "Qwen/Qwen2.5-7B-Instruct": [18],
            "Qwen/Qwen2.5-Coder-7B-Instruct": [18],
            "meta-llama/Llama-3.1-8B-Instruct": [15],
            "codellama/CodeLlama-7b-Instruct-hf": [13],
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": [19],
            "deepseek-ai/deepseek-coder-6.7b-instruct": [16]
        }

colonne_sicuro =["Steering_Forza (S_Steering - V_base)",
    "Steering_Overshooting (S_Steering - S_base)",
    "Baseline_Vanilla (S_base - V_base)"]

colonne_vulnerabile = ["Steering_Overshooting (V_base - V_Steering)",
                       "Sbase-Vsteering (S_base - V_Steering)",
                       "Baseline_Vanilla (S_base - V_base)"
                       ]

for model_name, config in models_config.items():

    lati_sicuro = []
    lati_vulnerabile = []

    layer_locus = layer_per_modello.get(model_name)[0]

    df_filtrato = df[(df['MODELLO'] == model_name) & (df['LAYER'] == layer_locus)]

    lati_sicuro = []
    for colonna in colonne_sicuro:
        valore = df_filtrato[colonna].values[0]
        lati_sicuro.append(valore)
        print(f"Estratto Sicuro -> {colonna}: {valore}")

    lati_vulnerabile = []
    for colonna in colonne_vulnerabile:
        valore = df_filtrato[colonna].values[0]
        lati_vulnerabile.append(valore)
        print(f"Estratto Vulnerabile -> {colonna}: {valore}")

    compara_due_triangoli(lati_sicuro, lati_vulnerabile, nome=model_name, nome_t1="Sicuro", nome_t2="Vulnerabile")
    nome_file_safe = model_name.replace('/', '_')


    percorso_salvataggio = f"L2_COMPLETI/Steering_{nome_file_safe}.png"
        
    plt.savefig(
        percorso_salvataggio, 
        dpi=300,               
        bbox_inches='tight',   
        format='png'           
    )
    print(f"Grafico salvato con successo in: {percorso_salvataggio}")
        