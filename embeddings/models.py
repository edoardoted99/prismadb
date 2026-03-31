from django.db import models
from project.constants import DOC_PENDING, DOC_DONE, DOC_ERROR


class Dataset(models.Model):
    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(blank=True)
    model_name = models.CharField(max_length=200, default="nomic-embed-text:latest")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    def total_docs(self):
        return self.documents.count()

    def done_docs(self):
        return self.documents.filter(status=DOC_DONE).count()

    def error_docs(self):
        return self.documents.filter(status=DOC_ERROR).count()

    def pending_docs(self):
        return self.documents.filter(status=DOC_PENDING).count()

    def progress_percent(self):
        total = self.total_docs()
        if total == 0:
            return 0
        return round(self.done_docs() * 100.0 / total, 1)


class Document(models.Model):
    STATUS_CHOICES = [
        (DOC_PENDING, "Pending"),
        (DOC_DONE, "Done"),
        (DOC_ERROR, "Error"),
    ]

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="documents")
    external_id = models.CharField(max_length=255)
    text = models.TextField()
    embedding = models.JSONField(null=True, blank=True)
    opensearch_id = models.CharField(max_length=255, blank=True, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=DOC_PENDING)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("dataset", "external_id")

    def __str__(self):
        return f"{self.dataset.name} :: {self.external_id}"
