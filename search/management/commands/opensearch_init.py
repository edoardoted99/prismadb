from django.core.management.base import BaseCommand
from embeddings.models import Dataset
from sae.models import SAERun
from search.indices import create_document_index, create_feature_index, get_embedding_dim


class Command(BaseCommand):
    help = "Create OpenSearch indices for existing datasets and SAE runs"

    def handle(self, *args, **options):
        # Document indices
        for dataset in Dataset.objects.all():
            dim = get_embedding_dim(dataset.model_name)
            self.stdout.write(
                f"Creating document index for dataset {dataset.id} "
                f"({dataset.name}), dim={dim}"
            )
            create_document_index(dataset.id, dim)

        # Feature indices
        for run in SAERun.objects.filter(status='completed'):
            self.stdout.write(f"Creating feature index for run {run.id}")
            create_feature_index(run.id)

        self.stdout.write(self.style.SUCCESS("OpenSearch indices created."))
