import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import models_config
import pandas as pd

#### BLU SICURO
#### ROSSO VULNERABILE

# --- MAGIA DI SEABORN ---
# Usa lo stile 'whitegrid' (sfondo bianco, griglia chiara)
# Il context 'talk' ingrandisce proporzionalmente TUTTO: testi, linee, punti.
sns.set_theme(style="whitegrid", context="talk")

def calcola_coordinate_e_angoli(a, b, c):
    if a + b <= c or a + c <= b or b + c <= a:
        raise ValueError(f"I lati {a}, {b}, {c} non formano un triangolo valido!")

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
    #aggiungi_etichette_lati(coords_1[0], coords_1[1], lati_t1, colore=colore_1, offset_y=-0.15)

    # Disegno Triangolo 2
    plt.plot(coords_2[0], coords_2[1], marker='s', color=colore_2, linestyle='--', label=nome_t2)
    plt.fill(coords_2[0], coords_2[1], color=colore_2, alpha=0.2)
    #aggiungi_etichette_lati(coords_2[0], coords_2[1], lati_t2, colore=colore_2, offset_y=0.15)

    plt.axis('equal') 
    
    # Seaborn scala già i font, dobbiamo solo inserire i testi
    #plt.legend(loc="upper right")
    
    # Aggiungo un titolo con un po' di margine inferiore (pad)
    plt.title(f"{nome}\n Distanze L2", fontweight='bold', pad=20)
    plt.xlabel("L2 Proiettata (Asse X)", labelpad=15)
    plt.ylabel("L2 Proiettata (Asse Y)", labelpad=15)

    # Togliamo i bordi "duri" (spines) tipici di seaborn per un look ancora più pulito
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()


df = pd.read_csv("Distanze_L2_values.csv")

layer_per_modello = {
            "Qwen/Qwen2.5-7B-Instruct": [26],
            "Qwen/Qwen2.5-Coder-7B-Instruct": [26],
            "meta-llama/Llama-3.1-8B-Instruct": [30],
            "codellama/CodeLlama-7b-Instruct-hf": [30],
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": [26],
            "deepseek-ai/deepseek-coder-6.7b-instruct": [30]
        }

colonne_sicuro =["AA_Forza (S_AA - V_base)",
    "AA_Overshooting (S_AA - S_base)",
    "Baseline_Vanilla (S_base - V_base)"]

colonne_vulnerabile = ["AA (V_base - V_AA)",
                       "AA (S_base - V_AA)",
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

    # 2. Riempiamo la lista per il secondo triangolo
    lati_vulnerabile = []
    for colonna in colonne_vulnerabile:
        valore = df_filtrato[colonna].values[0]
        lati_vulnerabile.append(valore)
        print(f"Estratto Vulnerabile -> {colonna}: {valore}")

    compara_due_triangoli(lati_sicuro, lati_vulnerabile, nome=model_name, nome_t1="Sicuro", nome_t2="Vulnerabile")
    nome_file_safe = model_name.replace('/', '_')

    percorso_salvataggio = f"Grafici_finali/Last layer/Advanced adversarial/{nome_file_safe}.png"
        
    plt.savefig(
        percorso_salvataggio, 
        dpi=300,               # 300 DPI è lo standard per l'alta risoluzione su carta
        bbox_inches='tight',   # Evita che il titolo o le etichette degli assi vengano tagliati ai bordi
        format='png'           # Puoi usare anche 'pdf' se scrivi la tesi in LaTeX per avere grafica vettoriale!
    )
    print(f"Grafico salvato con successo in: {percorso_salvataggio}")
        