# sae/forms.py
from django import forms
from .models import SAERun

class SAERunForm(forms.ModelForm):
    class Meta:
        model = SAERun
        fields = ['dataset', 'expansion_factor', 'k_sparsity', 'alpha_aux', 
                  'learning_rate', 'batch_size', 'epochs']
        widgets = {
            'dataset': forms.Select(attrs={'class': 'form-select'}),
            'expansion_factor': forms.NumberInput(attrs={'class': 'form-control'}),
            'k_sparsity': forms.NumberInput(attrs={'class': 'form-control'}),
            'alpha_aux': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'learning_rate': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.0001'}),
            'batch_size': forms.NumberInput(attrs={'class': 'form-control'}),
            'epochs': forms.NumberInput(attrs={'class': 'form-control'}),
        }