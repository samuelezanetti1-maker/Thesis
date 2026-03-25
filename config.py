# config.py
import pandas as pd
import torch
from transformers import BitsAndBytesConfig


config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16, 
    bnb_4bit_use_double_quant=True,       
    bnb_4bit_quant_type="nf4"             
)


models_config = {
    "Qwen/Qwen2.5-3B-Instruct": {"dtype": torch.float16},
    "Qwen/Qwen2.5-Coder-3B-Instruct": {"dtype": torch.float16},

    "meta-llama/Llama-3.2-3B-Instruct": {"dtype": torch.float16},
    #"codellama/CodeLlama-7b-Instruct-hf": {"quantization_config": config_4bit},  #Richiede troppa VRAM, non riesco a farlo funzionare

    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {"dtype": torch.float16},
    #"deepseek-ai/deepseek-coder-1.3b-instruct": {"dtype": torch.float16},
    
}


path = "/home/samuele/Desktop/dataset_tesi.csv"
df_globale = pd.read_csv(path)
df_globale = df_globale.head(10)  




