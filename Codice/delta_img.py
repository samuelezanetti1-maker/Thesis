import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", context="talk")
os.makedirs("Grafici_Logit_Lens", exist_ok=True)

# 1. Carica i due CSV con le nuove colonne "Base_"
df_succ = pd.read_csv("CSV tesi/Logit_Lens/Risultati_Contrastive_All_Layers.csv")
df_fail = pd.read_csv("CSV tesi/Logit_Lens/Risultati_Contrastive_All_Layers_fallimenti.csv")


# Dizionario per accoppiare le colonne Delta con le rispettive Basi
attacchi = {
    "Advanced Adversarial": {"delta": "Delta_AA", "base": "Base_AA"},
    "Prompt Injection": {"delta": "Delta_PI", "base": "Base_PI"},
    "Adversarial Perturbation": {"delta": "Delta_AP", "base": "Base_AP"},
    "Steering": {"delta": "Delta_Steer", "base": "Base_Steer"},
}

modelli = df_succ['MODELLO'].unique()

print("Avvio generazione grafici assoluti...\n" + "="*50)

for modello in modelli:
    print(f"\nElaborazione modello: {modello}")
    nome_file_safe = modello.replace('/', '_')
    
    dati_succ_mod = df_succ[df_succ['MODELLO'] == modello].sort_values('LAYER')
    dati_fail_mod = df_fail[df_fail['MODELLO'] == modello].sort_values('LAYER')

    
    if dati_succ_mod.empty or dati_fail_mod.empty:
        print(f"  [!] Dati mancanti per {modello}. Salto.")
        continue

        
    layers = dati_succ_mod['LAYER'].values
    
    for nome_attacco, colonne in attacchi.items():
        col_delta = colonne["delta"]
        col_base = colonne["base"]
        
        print(f"  -> Generazione: {nome_attacco}")
        
        # Estrazione Dati: SUCCESSI (Inganno)
        base_succ = dati_succ_mod[col_base].values
        delta_succ = dati_succ_mod[col_delta].values
        finale_succ = base_succ + delta_succ  # L'Equazione Fondamentale!

        # Estrazione Dati: FALLIMENTI (Resilienza)
        base_fail = dati_fail_mod[col_base].values
        delta_fail = dati_fail_mod[col_delta].values
        finale_fail = base_fail + delta_fail  # L'Equazione Fondamentale!

        # =========================================================
        # Creazione del Grafico (SIDE-BY-SIDE / DUE PANNELLI)
        # =========================================================
        # Creiamo una figura larga con due sottomenu (ax1, ax2). sharey=True è fondamentale!
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

        # Colori Originali
        c_ingannato = "purple"       
        c_ingannato_base = "#BCA9D1"  
        c_resiliente = "darkorange"      
        c_resiliente_base = "#F5CBA7" 

        # Titolo Globale per tutta l'immagine
        fig.suptitle(f"{nome_attacco}\n{modello}", 
                     fontweight='bold', fontsize=16, y=0.98)

        # ---------------------------------------------------------
        # PANNELLO 1 (SINISTRA): ATTACCO FALLITO (ARANCIONE)
        # ---------------------------------------------------------
        ax1.axhline(0, color='#2c3e50', linestyle='-', linewidth=1.5, alpha=0.8, zorder=1)
        
        ax1.plot(layers, base_fail, color=c_resiliente_base, linestyle=':', linewidth=2, 
                 label='Baseline')
        ax1.plot(layers, finale_fail, color=c_resiliente, linewidth=2.5, 
                 marker='o', markersize=6, markeredgecolor='white', markeredgewidth=1.2, zorder=5,
                 label='Esito: True')
        ax1.fill_between(layers, base_fail, finale_fail, color=c_resiliente, alpha=0.08, linewidth=0)

        # Estetica Pannello Sinistro
        ax1.set_title("Attacco Fallito", fontweight='bold', color=c_resiliente, pad=15)
        ax1.set_xlabel("Profondità della Rete (Layer)", labelpad=15, fontweight='bold')
        ax1.set_ylabel("Valore Assoluto Logit (True - False)", labelpad=15, fontweight='bold')
        ax1.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        ax1.set_facecolor('#fafafa')
        ax1.set_xlim(0, max(layers))
        ax1.set_xticks(range(0, max(layers)+1, 2))
        ax1.legend(loc='best', framealpha=0.9)

        # ---------------------------------------------------------
        # PANNELLO 2 (DESTRA): ATTACCO RIUSCITO (VIOLA)
        # ---------------------------------------------------------
        ax2.axhline(0, color='#2c3e50', linestyle='-', linewidth=1.5, alpha=0.8, zorder=1)
        
        ax2.plot(layers, base_succ, color=c_ingannato_base, linestyle=':', linewidth=2, 
                 label='Baseline')
        ax2.plot(layers, finale_succ, color=c_ingannato, linewidth=2.5, 
                 marker='D', markersize=5, markeredgecolor='white', markeredgewidth=1.2, zorder=6,
                 label='Esito: False')
        ax2.fill_between(layers, base_succ, finale_succ, color=c_ingannato, alpha=0.08, linewidth=0)

        # Estetica Pannello Destro
        ax2.set_title("Attacco Riuscito", fontweight='bold', color=c_ingannato, pad=15)
        ax2.set_xlabel("Profondità della Rete (Layer)", labelpad=15, fontweight='bold')
        # L'asse Y è condiviso, quindi non c'è bisogno di rimettere la label a destra
        ax2.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        ax2.set_facecolor('#fafafa')
        ax2.set_xlim(0, max(layers))
        ax2.set_xticks(range(0, max(layers)+1, 2))
        ax2.legend(loc='best', framealpha=0.9)

        # =========================================================
        # Salvataggio
        # =========================================================
        sns.despine(ax=ax1, bottom=True, left=True)
        sns.despine(ax=ax2, bottom=True, left=True)
        plt.tight_layout(rect=[0, 0, 1, 0.93])


        percorso_salvataggio = f"Grafici_Logit_Lens/Assoluto_{col_delta}_{nome_file_safe}.png"
        plt.savefig(percorso_salvataggio, dpi=300)
        plt.close()

print("\n" + "="*50 + "\nGenerazione completata! Controlla la cartella /Grafici_Logit_Lens")