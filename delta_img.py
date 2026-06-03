import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", context="talk")
os.makedirs("Grafici_Logit_Lens", exist_ok=True)

# 1. Carica i due CSV
df_succ = pd.read_csv("CSV tesi/Logit_Lens/Risultati_Contrastive_All_Layers.csv")
df_fail = pd.read_csv("CSV tesi/Logit_Lens/Risultati_Contrastive_All_Layers_fallimenti.csv")

# Mappatura delle colonne degli attacchi ai loro nomi formali per i titoli
attacchi = {
    "Delta_AA": "Advanced Adversarial",
    "Delta_PI": "Prompt Injection",
    "Delta_AP": "Adversarial Perturbation"
}

# Ottieni la lista di tutti i modelli unici presenti nel CSV
modelli = df_succ['MODELLO'].unique()

print("Avvio generazione massiva dei grafici...\n" + "="*50)

# Ciclo sui Modelli
for modello in modelli:
    print(f"\nElaborazione modello: {modello}")
    nome_file_safe = modello.replace('/', '_')
    
    # Filtra i dati e mettili in ordine per layer
    dati_succ_mod = df_succ[df_succ['MODELLO'] == modello].sort_values('LAYER')
    dati_fail_mod = df_fail[df_fail['MODELLO'] == modello].sort_values('LAYER')
    
    # Controllo di sicurezza: se il modello non ha dati in uno dei due CSV, saltalo
    if dati_succ_mod.empty or dati_fail_mod.empty:
        print(f"  [!] Dati mancanti per {modello}. Salto al prossimo.")
        continue
        
    layers = dati_succ_mod['LAYER'].values
    
    # Ciclo sui 3 Delta (Attacchi)
    for colonna_target, nome_attacco in attacchi.items():
        print(f"  -> Generazione tracciato: {nome_attacco} ({colonna_target})")
        
        # Estrai i valori Y per questo specifico attacco
        y_succ = dati_succ_mod[colonna_target].values
        y_fail = dati_fail_mod[colonna_target].values

        # =========================================================
        # Creazione del Grafico
        # =========================================================
        plt.figure(figsize=(12, 7))

        # La "Soglia del Dubbio" (Zero netto)
        plt.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Soglia Neutrale (0.0)')

        # Traiettoria Attacco Fallito (Resilienza)
        plt.plot(layers, y_fail, color='orange', linewidth=3, marker='o', 
                 markersize=6, label='Attacco Fallito')

        # Traiettoria Attacco Riuscito (Inganno)
        plt.plot(layers, y_succ, color='purple', linewidth=3, marker='s', 
                 markersize=6, label='Attacco Riuscito')

        # Coloriamo la differenza (Il "Gap di Resilienza")
        plt.fill_between(layers, y_succ, y_fail, color='gray', alpha=0.2, 
                         label='Gap di Forza Semantica')

        # Estetica Accademica
        plt.title(f"Tracciato Cognitivo Logit Lens: {nome_attacco}\n{modello}", 
                  fontweight='bold', pad=20)
        plt.xlabel("Profondità della Rete (Layer)", labelpad=15)
        plt.ylabel(r"$\Delta$ Logit (True - False)", labelpad=15)

        # Limiti e Ticks
        plt.xlim(0, max(layers))
        plt.xticks(range(0, max(layers)+1, 2))

        # Legenda e salvataggio
        plt.legend(loc='lower left', framealpha=0.9)
        plt.tight_layout()

        # Salvataggio dinamico basato su modello e attacco
        percorso_salvataggio = f"Grafici_Logit_Lens/Tracciato_{colonna_target}_{nome_file_safe}.png"
        plt.savefig(percorso_salvataggio, dpi=300)
        plt.close()

print("\n" + "="*50 + "\nGenerazione completata! Tutti i grafici sono in /Grafici_Logit_Lens")