# explorer/llm_utils.py
import json
import re
import time

import requests
from django.conf import settings

from project.utils import get_setting


def _ollama_base():
    return get_setting('ollama_base_url')


def get_ollama_models():
    """Returns all Ollama model names (for LLM chat/interpretation)."""
    try:
        response = requests.get(f"{_ollama_base()}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [m['name'] for m in data.get('models', [])]
    except Exception as e:
        print(f"Ollama Error: {e}")
    return []


def get_ollama_embedding_models():
    """
    Returns only Ollama models that support /api/embed.
    Checks name and family/families for embedding-related keywords.
    """
    EMBEDDING_KEYWORDS = {'embed', 'bert', 'bge', 'e5', 'gte'}
    try:
        response = requests.get(f"{_ollama_base()}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            embedding_models = []
            for m in data.get('models', []):
                name = m.get('name', '').lower()
                details = m.get('details', {})
                family = details.get('family', '').lower()
                families = [f.lower() for f in details.get('families', []) or []]

                searchable = name + ' ' + family + ' ' + ' '.join(families)
                if any(kw in searchable for kw in EMBEDDING_KEYWORDS):
                    embedding_models.append(m['name'])

            return embedding_models
    except Exception as e:
        print(f"Ollama Error: {e}")
    return []


# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a meticulous researcher investigating a specific neuron in a language model. Your task is to determine what behavior this neuron is responsible for: what concepts, topics, or linguistic features does it activate on?

INPUT DESCRIPTION: You will receive two inputs: 1) Maximum Activation Examples and 2) Zero Activation Examples.
1. You will be given several text examples that activate the neuron, along with a number indicating how strongly it was activated. This means there is some feature, concept, or pattern in this text that 'excites' this neuron.
2. You will also be given several text examples that do NOT activate the neuron. This means the feature or concept is not present in these texts.

OUTPUT DESCRIPTION: Given the inputs provided, complete the following tasks.
1. Based on the MAXIMUM ACTIVATION EXAMPLES, list potential topics, concepts, themes, and features they have in common. Be specific. You may need to look at different levels of granularity. List as many as possible. Give greater weight to concepts more prominent in higher-activation examples.
2. Based on the zero activation examples, systematically exclude any topic/concept/feature listed above that also appears in the zero activation examples.
3. Based on the two previous steps, perform a thorough analysis of which feature, concept, or topic, at which level of granularity, is likely to activate this neuron. Use Occam's razor, as long as it fits the evidence provided. Be highly rational and analytical.
4. Based on step 3, summarize this concept in 1-8 words, in the form FINAL: <explanation>. Do NOT return anything after these 1-8 words.

Respond EXCLUSIVELY with valid JSON: {'label': '...', 'description': '...'}"""


# Predictor prompt template (Paper §3.1)
# The Predictor evaluates interpretability by predicting feature activations on unseen text.
PREDICTOR_PROMPT_TEMPLATE = """You are evaluating whether a specific neuron in a language model would activate on a given text.

The neuron has been labeled as: "{label}"

Given the text below, predict your confidence that this neuron would activate on it.
Express your confidence as a single score from -1 to +1:
- +1 means absolute certainty the neuron WOULD activate
- 0 means completely uncertain
- -1 means absolute certainty the neuron would NOT activate

Respond EXCLUSIVELY with valid JSON: {{"score": <number between -1 and 1>}}"""


def get_ollama_response(user_message, system_message, model="qwen2.5:14b",
                        base_url=None, temperature=0.2, retries=2):
    if base_url is None:
        base_url = get_setting('ollama_base_url')
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
