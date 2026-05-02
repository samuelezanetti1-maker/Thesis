import matplotlib.pyplot as plt
import numpy as np

def calcola_coordinate_e_angoli(a, b, c):
    """
    Data la lunghezza dei 3 lati, restituisce le coordinate dei vertici
    (per matplotlib) e i 3 angoli interni in gradi.
    Il lato 'c' viene appoggiato sull'asse X.
    """
    # Controllo validità triangolo
    if a + b <= c or a + c <= b or b + c <= a:
        raise ValueError(f"I lati {a}, {b}, {c} non formano un triangolo valido!")

    # 1. Calcolo Coordinate
    x_A, y_A = 0, 0
    x_B, y_B = c, 0
    x_C = (c**2 + b**2 - a**2) / (2 * c)
    # Uso abs() per evitare problemi di precisione in virgola mobile di Python vicini allo 0
    y_C = np.sqrt(abs(b**2 - x_C**2)) 

    # 2. Calcolo Angoli con il Teorema del Coseno
    # L'angolo A è opposto al lato a, l'angolo B opposto a b, ecc.
    angolo_A = np.degrees(np.arccos((b**2 + c**2 - a**2) / (2 * b * c)))
    angolo_B = np.degrees(np.arccos((a**2 + c**2 - b**2) / (2 * a * c)))
    angolo_C = 180.0 - angolo_A - angolo_B # La somma degli angoli interni è sempre 180

    x_coords = [x_A, x_B, x_C, x_A]
    y_coords = [y_A, y_B, y_C, y_A]

    return (x_coords, y_coords), (angolo_A, angolo_B, angolo_C)

def compara_due_triangoli(lati_t1, lati_t2, nome_t1="Triangolo 1", nome_t2="Triangolo 2"):
    """
    Disegna due triangoli sovrapposti per confronto e stampa i loro angoli.
    lati_t1 e lati_t2 devono essere tuple di 3 numeri: (a, b, c)
    """
    # Estraggo coordinate e angoli
    coords_1, angoli_1 = calcola_coordinate_e_angoli(*lati_t1)
    coords_2, angoli_2 = calcola_coordinate_e_angoli(*lati_t2)

    # --- STAMPA DEI RISULTATI IN CONSOLE ---
    print(f"--- {nome_t1.upper()} ---")
    print(f"Lati: a={lati_t1[0]}, b={lati_t1[1]}, c={lati_t1[2]}")
    print(f"Angolo opposto ad 'a': {angoli_1[0]:.2f}°")
    print(f"Angolo opposto a  'b': {angoli_1[1]:.2f}°")
    print(f"Angolo opposto a  'c': {angoli_1[2]:.2f}°\n")

    print(f"--- {nome_t2.upper()} ---")
    print(f"Lati: a={lati_t2[0]}, b={lati_t2[1]}, c={lati_t2[2]}")
    print(f"Angolo opposto ad 'a': {angoli_2[0]:.2f}°")
    print(f"Angolo opposto a  'b': {angoli_2[1]:.2f}°")
    print(f"Angolo opposto a  'c': {angoli_2[2]:.2f}°\n")

    # --- PLOTTING ---
    plt.figure(figsize=(8, 6))

    # Disegno Triangolo 1 (es. Blu)
    plt.plot(coords_1[0], coords_1[1], marker='o', color='blue', linewidth=2, label=nome_t1)
    plt.fill(coords_1[0], coords_1[1], color='blue', alpha=0.3)

    # Disegno Triangolo 2 (es. Rosso)
    plt.plot(coords_2[0], coords_2[1], marker='s', color='red', linewidth=2, linestyle='--', label=nome_t2)
    plt.fill(coords_2[0], coords_2[1], color='red', alpha=0.3)

    # Impostazioni grafiche
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axis('equal') # Mantiene le proporzioni!
    plt.legend(loc="upper right")
    plt.title("Confronto tra due triangoli (es. Spostamento Centroidi)")
    plt.xlabel("Distanza L2 (Asse X)")
    plt.ylabel("Distanza L2 (Asse Y)")

    plt.show()

# ==========================================
# INSERISCI QUI I DATI DEI TUOI ESPERIMENTI
# ==========================================
# Esempio: i tre lati sono le tre distanze L2 tra i centroidi
lati_baseline = (11.266, 26.48, 21.61) 
lati_attacco  = (11.266, 26.62, 22.77) 

compara_due_triangoli(lati_baseline, lati_attacco, nome_t1="Baseline (Sicuro/Vuln)", nome_t2="Sotto Attacco")
