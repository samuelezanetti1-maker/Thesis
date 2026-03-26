# config.py
import os

import pandas as pd
import torch
from transformers import BitsAndBytesConfig

os.environ["HF_TOKEN"] = "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"
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_HUB_CACHE"] = "/scratch_share/bislab/HF_HUB_CACHE/"


config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16, 
    bnb_4bit_use_double_quant=True,       
    bnb_4bit_quant_type="nf4"             
)

### Versione casalinga
# models_config = {
#     "Qwen/Qwen2.5-3B-Instruct": {"dtype": torch.float16},
#     "Qwen/Qwen2.5-Coder-3B-Instruct": {"dtype": torch.float16},

#     "meta-llama/Llama-3.2-3B-Instruct": {"dtype": torch.float16},
#     #"codellama/CodeLlama-7b-Instruct-hf": {"quantization_config": config_4bit},  #Richiede troppa VRAM, non riesco a farlo funzionare

#     "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {"dtype": torch.float16},
#     #"deepseek-ai/deepseek-coder-1.3b-instruct": {"dtype": torch.float16},
    
# }

### Versione cluster
models_config = {
    # Famiglia Qwen (Alibaba)
    "Qwen/Qwen2.5-7B-Instruct": {"dtype": torch.float16},
    "Qwen/Qwen2.5-Coder-7B-Instruct": {"dtype": torch.float16},
    
    # Famiglia Llama (Meta)
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {"dtype": torch.float16},
    "codellama/CodeLlama-7b-Instruct-hf": {"dtype": torch.float16},
    
    # Famiglia DeepSeek
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {"dtype": torch.float16},
    "deepseek-ai/deepseek-coder-6.7b-instruct": {"dtype": torch.float16}
}


path = "CSV tesi/Dataset/dataset_tesi.csv"
df_globale = pd.read_csv(path)
df_globale = df_globale.sample(n=100, random_state=42).copy()
#df_globale = df_globale.head(10)  




