#utils.py
import re

# 1. Estrae la predizione del modello e la converte in testo
def evaluate_response(response):
    response = str(response).lower()
    
    # 1. Pulizia preventiva degli artefatti tipici di tokenizer (DeepSeek ecc.)
    response = response.replace('ġ', ' ').replace('ċ', ' ').replace('\n', ' ')
    
    # 2. Ricerca con RegEx: Cerca 'final_verdict' seguito da qualsiasi 
    # carattere (spazi, due punti, ecc.) fino a trovare 'true' o 'false'
    match = re.search(r'final_verdict.*?(\btrue\b|\bfalse\b)', response)
    
    if match:
        verdetto = match.group(1)
        if verdetto == 'true':
            return "Vulnerabile"
        elif verdetto == 'false':
            return "Sicuro"
            
    return "Errore_Formato"

# 2. Converte il target reale in testo
def traduci_target_reale(target):
    if str(target).lower() == 'true':
        return "Vulnerabile"
    else:
        return "Sicuro"
    
# 3. Funzione per identificare i veri positivi (TP) in un DataFrame
def is_true_positive(df):
    df_TP = df[(df['target_vero'] == 'Vulnerabile') & (df['target_predetto'] == 'Vulnerabile')]
    return df_TP

# 4. Funzione per identificare i falsi negativi (TN) in un DataFrame
def is_true_negative(df):
    df_TN = df[(df['target_vero'] == 'Sicuro') & (df['target_predetto'] == 'Sicuro')]
    return df_TN

# 5. Estrae la predizione, specifico per logit bias
def evaluate_response_logit(risposta_grezza):
    if not isinstance(risposta_grezza, str) or not risposta_grezza.strip():
        return "Non Classificato"
    risposta_lower = risposta_grezza.lower()

    keyword_sicuro = ["false", "secure", "sicuro", "suro", "sicic", "falseuroic"]
    keyword_vulnerabile = ["true", "vulnerable", "vulnerabile", "vuln"]

    # Contatori di frequenza
    score_sicuro = 0
    score_vulnerabile = 0

    # Contiamo quante volte appare ogni frammento
    for kw in keyword_sicuro:
        score_sicuro += risposta_lower.count(kw)

    for kw in keyword_vulnerabile:
        score_vulnerabile += risposta_lower.count(kw)

    if score_sicuro == 0 and score_vulnerabile == 0:
        return "Babbling Estremo (Non Classificato)"

    # Se chiara prevalenza di token "Sicuro/False"
    if score_sicuro > score_vulnerabile:
        return "Sicuro"
        
    # Se chiara prevalenza di token "Vulnerabile/True"
    elif score_vulnerabile > score_sicuro:
        return "Vulnerabile"
        
    # 4. Caso di pareggio guardo chi ha vinto il primissimo token
    else:
        inizio = risposta_lower.strip()
        if inizio.startswith("false") or inizio.startswith("sic") or inizio.startswith("sec"):
            return "Sicuro"
        elif inizio.startswith("true") or inizio.startswith("vuln"):
            return "Vulnerabile"
        else:
            return "Non Classificato"