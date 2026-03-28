from django import forms
from explorer.llm_utils import get_ollama_embedding_models


class UploadDatasetForm(forms.Form):
    name = forms.CharField(
        label="Nome dataset",
        max_length=255,
        help_text="Un nome unico per questo dataset (es. 'faq_it_maggio').",
    )
    description = forms.CharField(
        label="Descrizione",
        widget=forms.Textarea,
        required=False,
    )
    model_name = forms.ChoiceField(
        label="Modello embeddings",
        choices=[],  # populated dynamically from Ollama
    )
    file = forms.FileField(
        label="File JSON",
        help_text="Carica un file JSON con una lista di oggetti {id, text}.",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['model_name'].choices = self._get_model_choices()

    @staticmethod
    def _get_model_choices():
        models = get_ollama_embedding_models()
        if models:
            return [(m, m) for m in models]
        return [("", "-- No embedding models found (install one: ollama pull nomic-embed-text) --")]

    def clean_file(self):
        f = self.cleaned_data["file"]
        if not f.name.lower().endswith(".json"):
            raise forms.ValidationError("Il file deve essere un .json")
        return f
