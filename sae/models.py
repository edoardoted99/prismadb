# sae/models.py
from django.db import models
from embeddings.models import Dataset
from project.constants import RUN_QUEUED, RUN_RUNNING, RUN_COMPLETED, RUN_FAILED

class SAERun(models.Model):
    STATUS_CHOICES = [
        (RUN_QUEUED, 'Queued'),
        (RUN_RUNNING, 'Training'),
        (RUN_COMPLETED, 'Completed'),
        (RUN_FAILED, 'Failed'),
    ]

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="sae_runs")

    # --- Configuration Hyperparameters ---
    input_dim = models.IntegerField(help_text="Input embedding dimension (e.g. 768)")
    expansion_factor = models.IntegerField(default=4, help_text="Latent expansion factor (e.g. 4x)")
    k_sparsity = models.IntegerField(default=32, help_text="Top-K active neurons")

    
    
    # Loss & Training Params
    alpha_aux = models.FloatField(default=0.03, help_text="Auxiliary Loss Coefficient")
    learning_rate = models.FloatField(default=1e-4)
    batch_size = models.IntegerField(default=512)
    epochs = models.IntegerField(default=20)

    # --- Status & Results ---
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=RUN_QUEUED)
    error_message = models.TextField(blank=True)
    
    # Files
    weights_file = models.FileField(upload_to='sae_weights/', null=True, blank=True)
    sparsity_heatmap = models.ImageField(upload_to='sae_plots/', null=True, blank=True)
    
    # Matrix Analysis Heatmaps
    matrix_s_heatmap = models.ImageField(upload_to='sae_plots/', null=True, blank=True, help_text="Similarity Matrix (Decoder Weights)")
    matrix_c_heatmap = models.ImageField(upload_to='sae_plots/', null=True, blank=True, help_text="Co-occurrence Matrix")
    matrix_d_heatmap = models.ImageField(upload_to='sae_plots/', null=True, blank=True, help_text="Dense Activation Similarity Matrix")
    
    
    # Final Metrics
    final_loss = models.FloatField(null=True, blank=True)
    
    # History Log: list of dicts [{'epoch':1, 'loss':0.5}, ...]
    training_log = models.JSONField(default=list, blank=True) 

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Run #{self.id} [{self.dataset.name}] - {self.get_status_display()}"