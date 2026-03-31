# explorer/llm_utils.py
import json
import requests
import time
import re
from django.conf import settings

OLLAMA_BASE = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")




def get_ollama_models():
    """Returns all Ollama model names (for LLM chat/interpretation)."""
    try:
        response = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=0.5)
        if response.status_code == 200:
            data = response.json()
            return [m['name'] for m in data.get('models', [])]
    except Exception as e:
        print(f"Ollama Error: {e}")
    return []


def get_ollama_embedding_models():
    """
    Returns only Ollama models that support /api/embed.
    Filters by name containing 'embed' or family containing 'bert'.
    """
    try:
        response = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=0.5)
        if response.status_code == 200:
            data = response.json()
            embedding_models = []
            for m in data.get('models', []):
                name = m.get('name', '').lower()
                details = m.get('details', {})
                family = details.get('family', '').lower()

                if 'embed' in name or 'bert' in family:
                    embedding_models.append(m['name'])

            return embedding_models
    except Exception as e:
        print(f"Ollama Error: {e}")
    return []


# Prompt di default
DEFAULT_SYSTEM_PROMPT = """Sei un meticoloso ricercatore che sta conducendo un'importante indagine su un certo neurone in un modello linguistico addestrato su cartelle cliniche pediatriche. Il tuo compito è capire di quale comportamento è responsabile questo neurone: ovvero, su quali concetti clinici, sintomi, diagnosi, trattamenti o terminologie mediche specifiche si attiva questo neurone? Ecco come completerai il compito:

DESCRIZIONE INPUT: Ti verranno forniti due input: 1) Esempi di Massima Attivazione e 2) Esempi di Zero Attivazione.
1. Ti verranno forniti diversi esempi di testo che attivano il neurone, insieme a un numero che indica quanto è stato attivato. Questo significa che c'è qualche caratteristica, sintomo, patologia o concetto clinico in questo testo che 'eccita' questo neurone.
2. Ti verranno forniti anche diversi esempi di testo che non attivano il neurone. Questo significa che la caratteristica o il concetto non è presente in questi testi.

DESCRIZIONE OUTPUT: Dati gli input forniti, completa i seguenti compiti.
1. Basandoti sugli ESEMPI DI MASSIMA ATTIVAZIONE forniti, scrivi potenziali argomenti, concetti, temi, metodologie e caratteristiche che hanno in comune. Questi dovranno essere specifici - ricorda, tutto il testo proviene da cartelle cliniche pediatriche, quindi devono essere concetti altamente specifici della materia. Potresti dover guardare a diversi livelli di granularità. Elencane il più possibile. Dai maggior peso ai concetti più presenti/prominenti negli esempi con attivazioni più alte.
2. Basandoti sugli esempi di zero attivazione, escludi qualsiasi argomento/concetto/caratteristica elencato sopra che sia presente negli esempi di zero attivazione. Esamina sistematicamente la tua lista.
3. Basandoti sui due passaggi precedenti, esegui un'analisi approfondita di quale caratteristica, concetto o argomento, a quale livello di granularità, è probabile che attivi questo neurone. Usa il rasoio di Occam, purché si adatti alle prove fornite. Sii altamente razionale e analitico.
4. Basandoti sul passaggio 3, riassumi questo concetto in 1-8 parole, nella forma FINALE: <spiegazione>. NON restituire nulla dopo queste 1-8 parole.

Rispondi ESCLUSIVAMENTE con un JSON valido: {'label': '...', 'description': '...'}"""


def get_ollama_response(user_message, system_message, model="qwen2.5:14b",
                        base_url=None, temperature=0.2, retries=2):
    if base_url is None:
        base_url = getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
    """
    Invia una richiesta a Ollama gestendo errori, JSON parsing e parametri avanzati.
    """
    url = f"{base_url}/api/chat"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        "format": "json",
        "stream": False,
            "options": {
            "temperature": float(temperature),  # Parametro dinamico
            "num_ctx": 16384*8,     # Ridotto da 131k a 16k per evitare OOM/500 Errors
            "num_predict": 512    # Limite output
        }
    }

    for attempt in range(retries + 1):
        try:
            timeout = getattr(settings, 'EXPLORER_OLLAMA_TIMEOUT', 300)
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "")
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Tentativo di recupero JSON sporco
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    return json.loads(match.group())
                
        except Exception as e:
            print(f"[LLM Error] Attempt {attempt+1}/{retries+1}: {e}")
            time.sleep(2)
    
    return None