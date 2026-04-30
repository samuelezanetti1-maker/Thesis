import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configurazione stile Seaborn per un look pulito e moderno
sns.set_theme(style="whitegrid")

# Dati
angolo_deg = 76.62
L_base = 2.764      # baseline_vanilla
L_sbase = 8.23      # saa_sbase
L_vbase = 6.977     # saa_vbase

# Conversione angolo
theta = np.radians(angolo_deg)

# Coordinate dei vertici
Ax, Ay = 0, 0
Bx, By = L_base, 0
Cx, Cy = L_sbase * np.cos(theta), L_sbase * np.sin(theta)

# Creazione della figura
plt.figure(figsize=(10, 8))

# Palette di colori Seaborn (più eleganti dei colori base)
colors = sns.color_palette("bright")

# Disegno dei lati come linee
plt.plot([Ax, Bx], [Ay, By], color=colors[0], linewidth=3, label=f'Baseline Vanilla ({L_base})', marker='o')
plt.plot([Ax, Cx], [Ay, Cy], color=colors[3], linewidth=3, label=f'SAA SBase ({L_sbase})', marker='o')
plt.plot([Bx, Cx], [By, Cy], color=colors[2], linewidth=2, linestyle='--', label=f'SAA VBase ({L_vbase})', marker='o')

# Titolo e Label
plt.title("Rappresentazione Geometrica Steering", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Coordinata X", fontsize=11)
plt.ylabel("Coordinata Y", fontsize=11)

# Fondamentale: mantiene le proporzioni reali 1:1
plt.axis('equal') 

# Legenda stilizzata
plt.legend(frameon=True, facecolor='white', shadow=True)

# Annotazione dell'angolo nell'origine
plt.annotate(f"{angolo_deg}°", xy=(0.4, 0.4), fontsize=12, fontweight='bold', color="#444444")

# Rimuove le linee esterne del grafico (opzionale, per pulizia)
sns.despine(left=True, bottom=True)

plt.show()