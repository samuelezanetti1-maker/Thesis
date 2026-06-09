import pandas as pd

# 1. Carica il dataset
df = pd.read_csv('CSV tesi/risultati_attacco_pizza_perturbation.csv')

df['corretto'] = df['target_predetto_pizza'].str.strip().str.lower() == 'vulnerabile'

risultati = df.groupby('modello').agg(
    totale=('id_snippet', 'count'),
    corretti=('corretto', 'sum')
)


risultati['percentuale_errati (%)'] = (1 - (risultati['corretti'] / risultati['totale'])) * 100
#risultati['percentuale_corretti (%)'] = (risultati['corretti'] / risultati['totale']) * 100

print(risultati.round(2))