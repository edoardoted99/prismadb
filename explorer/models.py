# explorer/models.py
from django.db import models

from sae.models import SAERun


class SAEFeature(models.Model):
    run = models.ForeignKey(SAERun, on_delete=models.CASCADE, related_name="features")
    feature_index = models.IntegerField(db_index=True)

    # --- Cache dell'interpretazione "Attiva/Principale" (per la lista veloce) ---
    label = models.CharField(max_length=255, blank=True)
    description = models.TextField(blank=True)

    # Statistiche "Hard" (non cambiano con l'interpretazione)
    density = models.FloatField(null=True)
    max_activation = models.FloatField(null=True)
    mean_activation = models.FloatField(null=True)
    variance_activation = models.FloatField(null=True)

    # Questi sono i Top-K globali calcolati dallo script di Statistiche (Ground Truth)
    example_docs = models.JSONField(default=list, blank=True)

    correlated_features = models.JSONField(default=list, blank=True)
    co_occurring_features = models.JSONField(default=list, blank=True)
    activation_histogram = models.JSONField(default=dict, blank=True)

    # Collegamento all'interpretazione selezionata
    active_interpretation = models.OneToOneField(
        'Interpretation', null=True, blank=True,
        on_delete=models.SET_NULL, related_name='active_for_feature'
    )

    class Meta:
        unique_together = ('run', 'feature_index')
        ordering = ['-max_activation']

    def __str__(self):
        return f"[{self.run.id}] Feat {self.feature_index}: {self.label}"
class FeatureFamily(models.Model):
    """
    Rappresenta un gruppo gerarchico di feature (Sezione 4.2 del paper).
    """
    run = models.ForeignKey(SAERun, on_delete=models.CASCADE, related_name="families")

    # La feature "radice" (concetto generale, es. "Astronomia")
    parent_feature = models.ForeignKey(SAEFeature, on_delete=models.CASCADE, related_name="family_parent")

    # Le feature "figlie" (concetti specifici, es. "Pulsar", "Stelle Binarie")
    children_features = models.ManyToManyField(SAEFeature, related_name="family_children")

    # Metadati dell'algoritmo
    iteration = models.IntegerField(default=0, help_text="Iterazione dell'algoritmo in cui è stata trovata")
    size = models.IntegerField(default=0)

    # Interpretazione della famiglia (opzionale: il paper suggerisce di dare un nome alla famiglia intera)
    family_label = models.CharField(max_length=255, blank=True, help_text="Etichetta riassuntiva della famiglia")

    class Meta:
        ordering = ['iteration', '-size'] # Prima le famiglie più grandi e generali

    def __str__(self):
        return f"Family: {self.parent_feature.label} ({self.size} children)"


class Interpretation(models.Model):
    """
    Storico delle interpretazioni generate per una singola feature.
    """
    feature = models.ForeignKey(SAEFeature, on_delete=models.CASCADE, related_name="interpretations")

    # Risultato LLM
    label = models.CharField(max_length=255)
    description = models.TextField()

    # Configurazione usata
    llm_model = models.CharField(max_length=100)
    temperature = models.FloatField()
    system_prompt = models.TextField()

    # Contesto esatto passato all'LLM (Prompt Evidence)
    # Salveremo: {'positive': [...], 'negative': [...]}
    evidence_docs = models.JSONField(default=dict)

    # Predictor LLM scores (Paper §3.1)
    predictor_pearson = models.FloatField(null=True, blank=True,
        help_text="Pearson correlation between predicted and actual activations")
    predictor_f1 = models.FloatField(null=True, blank=True,
        help_text="F1 score of binary activation prediction")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at'] # Dal più recente

    def __str__(self):
        return f"{self.label} ({self.llm_model})"


class AppSetting(models.Model):
    """Key/value store for application configuration persisted in SQLite."""
    key = models.CharField(max_length=128, unique=True, db_index=True)
    value = models.TextField(default='')

    class Meta:
        ordering = ['key']

    def __str__(self):
        return f"{self.key} = {self.value}"
