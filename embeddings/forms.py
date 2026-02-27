from django import forms
from .models import Dataset


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
        choices=Dataset.MODEL_CHOICES,
        initial="medbit",
    )
    file = forms.FileField(
        label="File JSON",
        help_text="Carica un file JSON con una lista di oggetti {id, text}.",
    )

    def clean_file(self):
        f = self.cleaned_data["file"]
        if not f.name.lower().endswith(".json"):
            raise forms.ValidationError("Il file deve essere un .json")
        return f
