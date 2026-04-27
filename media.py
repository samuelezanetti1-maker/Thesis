from config import models_config
import os
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

file_attacchi = "CSV tesi/Isomorfismo_Steering.csv" 
df_attacchi = pd.read_csv(file_attacchi)

risultati_media = []

for model_name, config in models_config.items():
    
    df_mod = df_attacchi[df_attacchi['modello'] == model_name]
    if len(df_mod) == 0:
        continue

    nome_modello_pulito = model_name.replace("/", "_")
    print("\n" + "="*70)
    print(f" CALCOLO MEDIA ISOMORFISMO SU: {model_name} ({len(df_mod)} attacchi riusciti)")
    print("="*70)

    media_isomorfismo_cosine = df_mod['cos_sim_stesso_layer'].mean()
    risultati_media.append({
        "modello": model_name,
        "cosine_similarity_media": media_isomorfismo_cosine
    })

    media_isomorfismo_abs = df_mod['cos_sim_penultimo_layer'].mean()
    risultati_media.append({
        "modello": model_name,
        "penultimo layer_media": media_isomorfismo_abs
    })

    print(f"modello:{model_name}: media isomorfismo: {media_isomorfismo_abs}\n media cosine similarity: {media_isomorfismo_cosine}")

