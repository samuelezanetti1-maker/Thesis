from config import models_config
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/scratch_share/bislab/HF_HUB_CACHE/"

import pandas as pd
import torch
import gc
import random
import re

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import evaluate_response

df_modello_aa = pd.read_csv("CSV tesi/Split_Dataset_succ/advanced_solo_successi.csv")
df_modello_ap = pd.read_csv("CSV tesi/Split_Dataset_succ/ap_solo_successi.csv")
df_modello_pj = pd.read_csv("CSV tesi/Split_Dataset_succ/PJ_solo_successi.csv")

for model_name, config in models_config.items():
    print("\n" + "="*60)
    print(f"INIZIO valutazione con: {model_name}")
    print("="*60)