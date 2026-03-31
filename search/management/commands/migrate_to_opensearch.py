from django.core.management.base import BaseCommand

from embeddings.models import Dataset, Document
from explorer.models import SAEFeature
from sae.models import SAERun
from search.bulk_ops import (
    bulk_index_documents,
    bulk_index_features,
    bulk_update_embeddings,
)
from search.indices import (
    create_document_index,
    create_feature_index,
    get_embedding_dim,
)


class Command(BaseCommand):
    help = "Migrate existing data from SQLite to OpenSearch"

    def add_arguments(self, parser):
        parser.add_argument('--batch-size', type=int, default=500)

    def handle(self, *args, **options):
        batch_size = options['batch_size']

        # 1. Migrate Documents
        for dataset in Dataset.objects.all():
            dim = get_embedding_dim(dataset.model_name)
            create_document_index(dataset.id, dim)

            self.stdout.write(
                f"Migrating documents for dataset {dataset.id} ({dataset.name})..."
            )

            docs = Document.objects.filter(dataset=dataset)
            total = docs.count()

            doc_batch = []
            emb_batch = []

            for i, doc in enumerate(docs.iterator(chunk_size=batch_size)):
                doc_batch.append({
                    'django_id': doc.id,
                    'external_id': doc.external_id,
                    'text': doc.text,
                    'status': doc.status,
                })

                if doc.embedding and doc.status == 'done':
                    emb_batch.append((doc.id, doc.embedding))

                if len(doc_batch) >= batch_size:
                    bulk_index_documents(dataset.id, doc_batch)
                    doc_batch = []

                if len(emb_batch) >= batch_size:
                    bulk_update_embeddings(dataset.id, emb_batch)
                    emb_batch = []

                if (i + 1) % 1000 == 0:
                    self.stdout.write(f"  {i + 1}/{total} documents...")

            # Flush remaining
            if doc_batch:
                bulk_index_documents(dataset.id, doc_batch)
            if emb_batch:
                bulk_update_embeddings(dataset.id, emb_batch)

            self.stdout.write(self.style.SUCCESS(
                f"  Done: {total} documents migrated."
            ))

        # 2. Migrate Features
        for run in SAERun.objects.filter(status='completed'):
            create_feature_index(run.id)

            features = SAEFeature.objects.filter(run=run)
            total = features.count()
            self.stdout.write(f"Migrating features for run {run.id}...")

            feat_batch = []
            for feat in features.iterator(chunk_size=batch_size):
                feat_batch.append({
                    'django_id': feat.id,
                    'feature_index': feat.feature_index,
                    'label': feat.label or '',
                    'description': feat.description or '',
                    'density': feat.density,
                    'max_activation': feat.max_activation,
                    'mean_activation': feat.mean_activation,
                    'variance_activation': feat.variance_activation,
                    'example_docs': feat.example_docs,
                    'correlated_features': feat.correlated_features,
                    'co_occurring_features': feat.co_occurring_features,
                    'activation_histogram': feat.activation_histogram,
                })

                if len(feat_batch) >= batch_size:
                    bulk_index_features(run.id, feat_batch)
                    feat_batch = []

            if feat_batch:
                bulk_index_features(run.id, feat_batch)

            self.stdout.write(self.style.SUCCESS(
                f"  Done: {total} features migrated."
            ))

        self.stdout.write(self.style.SUCCESS("Migration complete!"))
