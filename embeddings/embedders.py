import torch
from transformers import AutoTokenizer, AutoModel

HF_MODELS = {
    "medbit": "IVN-RIN/medBIT",
    #"bio_bert_sentence": "pritamdeka/Sentence-BioBERT",
    "gte_multilingual": "Alibaba-NLP/gte-multilingual-base",
    "sbert_minilm": "sentence-transformers/all-MiniLM-L6-v2",
}

class HuggingFaceEmbedder:
    _instances = {}  # Cache
    DEFAULT_MAX_SEQ_LENGTH = 512  # fallback se il tokenizer non è informativo

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.hf_id = HF_MODELS[model_name]
        self.device = device
        self.tokenizer = None
        self.model = None
        self.max_seq_length = None  # diventa per-modello
        self._load()

    def _load(self):
        """
        Loads the model and tokenizer.
        Auto-detects CUDA or MPS (Apple Silicon) if available.
        """
        if self.tokenizer is None or self.model is None:
            print(f"Loading {self.model_name} ({self.hf_id})...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hf_id,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.hf_id,
                trust_remote_code=True
            )
            # >>> Imposta max_seq_length dinamica <<<
            # HuggingFace di solito popola tokenizer.model_max_length
            max_len = getattr(self.tokenizer, "model_max_length", None)

            # Alcuni tokenizer usano valori enormi come "no limit" (es. 100000...)
            if max_len is None or max_len > 100000:
                # Proviamo dal config se esiste
                config_max = getattr(getattr(self.model, "config", None),
                                     "max_position_embeddings",
                                     None)
                if config_max is not None and config_max > 0:
                    max_len = config_max
                else:
                    max_len = self.DEFAULT_MAX_SEQ_LENGTH

            self.max_seq_length = int(max_len)
            print(f"Max seq length for {self.model_name}: {self.max_seq_length}")

            if not self.device:
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"

            print(f"Model {self.model_name} loaded on device: {self.device}")
            self.model.to(self.device)
            self.model.eval()

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        # come prima
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if self.model is None:
            self._load()

        all_embeddings = []

        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

            # Numero di special tokens (di solito 2: CLS+SEP)
            num_special = 0
            if self.tokenizer.cls_token_id is not None:
                num_special += 1
            if self.tokenizer.sep_token_id is not None:
                num_special += 1

            # Se il modello non usa SEP/CLS, puoi adattare qui
            effective_window = self.max_seq_length - num_special
            if effective_window <= 0:
                raise ValueError(f"Invalid effective_window={effective_window} for model {self.model_name}")

            chunk_embeddings = []

            for i in range(0, len(tokens), effective_window):
                chunk_ids = tokens[i : i + effective_window]

                input_ids = chunk_ids
                # Aggiungi special tokens solo se esistono
                if self.tokenizer.cls_token_id is not None:
                    input_ids = [self.tokenizer.cls_token_id] + input_ids
                if self.tokenizer.sep_token_id is not None:
                    input_ids = input_ids + [self.tokenizer.sep_token_id]

                input_tensor = torch.tensor([input_ids]).to(self.device)
                attention_mask = torch.ones_like(input_tensor).to(self.device)

                with torch.no_grad():
                    outputs = self.model(input_ids=input_tensor, attention_mask=attention_mask)

                pooled_output = self._mean_pooling(outputs, attention_mask)
                chunk_embeddings.append(pooled_output)

            if chunk_embeddings:
                stacked_chunks = torch.cat(chunk_embeddings, dim=0)
                final_embedding = torch.mean(stacked_chunks, dim=0)
                all_embeddings.append(final_embedding.cpu().tolist())
            else:
                all_embeddings.append([])

        return all_embeddings



def get_embedder(model_name: str):
    """
    Factory function to retrieve the correct embedder class instance.
    """
    if model_name in HF_MODELS:
        # We return an INSTANCE now, not a class, to keep state (model loaded)
        # But the previous code expected a class with a static method or class method.
        # Let's adapt to return an object that has embed_texts method.
        # To avoid reloading the model every time, we can cache instances in the factory.
        
        if model_name not in HuggingFaceEmbedder._instances:
             HuggingFaceEmbedder._instances[model_name] = HuggingFaceEmbedder(model_name)
        
        return HuggingFaceEmbedder._instances[model_name]

    raise ValueError(f"Unsupported model: {model_name}")
