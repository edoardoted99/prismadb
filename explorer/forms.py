# explorer/forms.py
from django import forms

from sae.models import SAERun

from .llm_utils import DEFAULT_SYSTEM_PROMPT


class InterpretForm(forms.Form):
    run = forms.ModelChoiceField(
        queryset=SAERun.objects.filter(status='completed').order_by('-created_at'),
        label="Select Training Run",
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    n_features = forms.IntegerField(
        min_value=1, initial=20, #max_value=1000
        label="Features to Interpret",
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

    k_positive = forms.IntegerField(
        min_value=1, max_value=50, initial=6,
        label="Positive Examples (Top-K)",
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

    k_negative = forms.IntegerField(
        min_value=0, max_value=50, initial=4,
        label="Negative Examples (Random)",
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

    ollama_model = forms.ChoiceField(
        label="Ollama Model ID",
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .llm_utils import get_ollama_models
        models = get_ollama_models()
        # Create choices list: [(model_id, model_id), ...]
        choices = [(m, m) for m in models] if models else [('qwen2.5:14b', 'qwen2.5:14b (Default)')]
        self.fields['ollama_model'].choices = choices

    temperature = forms.FloatField(
        min_value=0.0, max_value=1.0, initial=0.2, step_size=0.1,
        label="Temperature",
        help_text="0.0 = Deterministic, 1.0 = Creative.",
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )


    system_prompt = forms.CharField(
        label="System Prompt (Persona)",
        initial=DEFAULT_SYSTEM_PROMPT,
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 8, 'style': 'font-size: 0.85rem;'})
    )
