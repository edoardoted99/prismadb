import logging
import threading

import torch
from django.shortcuts import get_object_or_404
from rest_framework import status, viewsets
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from embeddings.embedders import get_embedder
from embeddings.models import Dataset
from embeddings.services import generate_embeddings_for_dataset, ingest_json_and_create_dataset
from explorer.interpreter import (
    interpret_single_feature,
    load_sae_model,
    run_interpretation_pipeline,
)
from explorer.models import FeatureFamily, SAEFeature
from explorer.statistics import calculate_statistics_pipeline
from explorer.task_status import TASK_PROGRESS
from project.__version__ import __version__
from project.constants import DOC_DONE, RUN_COMPLETED, RUN_QUEUED, RUN_RUNNING
from sae.models import SAERun
from sae.modules import zscore_transform
from sae.trainer import train_sae_run

from .serializers import (
    DatasetSerializer,
    DocumentSerializer,
    FeatureFamilySerializer,
    HybridSearchRequestSerializer,
    InferenceRequestSerializer,
    InterpretationSerializer,
    SAEFeatureDetailSerializer,
    SAEFeatureSerializer,
    SAERunCreateSerializer,
    SAERunSerializer,
    SearchRequestSerializer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

class StandardPagination(PageNumberPagination):
    page_size = 50
    page_size_query_param = "page_size"
    max_page_size = 500


# ---------------------------------------------------------------------------
# Dataset endpoints
# ---------------------------------------------------------------------------

class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.all().order_by("-created_at")
    serializer_class = DatasetSerializer
    pagination_class = StandardPagination

    @action(detail=True, methods=["post"])
    def generate_embeddings(self, request, pk=None):
        """Start embedding generation (async daemon thread)."""
        dataset = self.get_object()
        if dataset.pending_docs() == 0:
            return Response(
                {"detail": "No pending documents to embed."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        t = threading.Thread(
            target=generate_embeddings_for_dataset,
            args=(dataset.pk,),
            daemon=True,
        )
        t.start()
        return Response({"status": "started", "dataset_id": dataset.pk})

    @action(detail=True, methods=["get"])
    def documents(self, request, pk=None):
        """List documents for this dataset."""
        dataset = self.get_object()
        qs = dataset.documents.all().order_by("id")
        page = self.paginate_queryset(qs)
        if page is not None:
            serializer = DocumentSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        serializer = DocumentSerializer(qs, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=["post"], url_path="upload")
    def upload_documents(self, request, pk=None):
        """Upload a JSON file of documents to an existing dataset.

        Expects multipart form with a 'file' field containing JSON
        in the format: [{"id": "...", "text": "..."}, ...]
        """
        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            return Response(
                {"detail": "No file provided."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not uploaded_file.name.endswith(".json"):
            return Response(
                {"detail": "Only .json files are accepted."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        # For upload to existing dataset, we reuse ingest logic
        # but with the dataset's existing model_name
        dataset = self.get_object()
        try:
            # We create a new dataset with ingest, but the user wanted to add to existing
            # For simplicity, this creates a new one — use POST /datasets/ for new datasets
            new_dataset = ingest_json_and_create_dataset(
                uploaded_file, dataset.name + "_upload", "", dataset.model_name
            )
            return Response(DatasetSerializer(new_dataset).data, status=status.HTTP_201_CREATED)
        except ValueError as e:
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    def create(self, request, *args, **kwargs):
        """Create a dataset by uploading a JSON file.

        Expects multipart form with: name, model_name, file, and optional description.
        """
        name = request.data.get("name")
        description = request.data.get("description", "")
        model_name = request.data.get("model_name")
        uploaded_file = request.FILES.get("file")

        if not all([name, model_name, uploaded_file]):
            return Response(
                {"detail": "name, model_name, and file are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            dataset = ingest_json_and_create_dataset(
                uploaded_file, name, description, model_name
            )
            return Response(DatasetSerializer(dataset).data, status=status.HTTP_201_CREATED)
        except ValueError as e:
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)


# ---------------------------------------------------------------------------
# SAE Run endpoints
# ---------------------------------------------------------------------------

class SAERunViewSet(viewsets.ModelViewSet):
    queryset = SAERun.objects.all().order_by("-created_at")
    serializer_class = SAERunSerializer
    pagination_class = StandardPagination

    def get_serializer_class(self):
        if self.action == "create":
            return SAERunCreateSerializer
        return SAERunSerializer

    def perform_create(self, serializer):
        dataset = serializer.validated_data["dataset"]
        first_doc = dataset.documents.filter(status=DOC_DONE).first()
        if first_doc and first_doc.embedding:
            input_dim = len(first_doc.embedding)
        else:
            input_dim = 0
        serializer.save(input_dim=input_dim, status=RUN_QUEUED)

    @action(detail=True, methods=["post"])
    def start_training(self, request, pk=None):
        """Start SAE training (async daemon thread)."""
        run = self.get_object()
        if run.status == RUN_RUNNING:
            return Response(
                {"detail": "Training is already running."},
                status=status.HTTP_409_CONFLICT,
            )
        t = threading.Thread(target=train_sae_run, args=(run.id,), daemon=True)
        t.start()
        return Response({"status": "started", "run_id": run.id})

    @action(detail=True, methods=["get"])
    def progress(self, request, pk=None):
        """Return training progress."""
        run = self.get_object()
        # Check thread-level progress
        thread_progress = None
        for tid, info in TASK_PROGRESS.items():
            if info.get("run_id") == run.id:
                thread_progress = info
                break
        return Response({
            "status": run.status,
            "training_log": run.training_log,
            "final_loss": run.final_loss,
            "thread_progress": thread_progress,
        })

    @action(detail=True, methods=["get"])
    def features(self, request, pk=None):
        """List features for this SAE run."""
        run = self.get_object()
        qs = run.features.all()

        # Optional search filter
        q = request.query_params.get("q")
        if q:
            qs = qs.filter(label__icontains=q)

        page = self.paginate_queryset(qs)
        if page is not None:
            serializer = SAEFeatureSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        serializer = SAEFeatureSerializer(qs, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=["get"], url_path=r"features/(?P<feature_index>\d+)")
    def feature_detail(self, request, pk=None, feature_index=None):
        """Get detail for a specific feature."""
        run = self.get_object()
        feature = get_object_or_404(SAEFeature, run=run, feature_index=feature_index)
        serializer = SAEFeatureDetailSerializer(feature)
        data = serializer.data

        # Include interpretation history
        interpretations = feature.interpretations.all()
        data["interpretations"] = InterpretationSerializer(interpretations, many=True).data
        return Response(data)

    @action(
        detail=True, methods=["post"],
        url_path=r"features/(?P<feature_index>\d+)/reinterpret",
    )
    def reinterpret_feature(self, request, pk=None, feature_index=None):
        """Trigger LLM re-interpretation of a feature."""
        run = self.get_object()
        feature = get_object_or_404(SAEFeature, run=run, feature_index=feature_index)

        model_name = request.data.get("model_name", "qwen2.5:14b")
        k_positive = int(request.data.get("k_positive", 6))
        k_negative = int(request.data.get("k_negative", 4))
        temperature = float(request.data.get("temperature", 0.2))
        prompt = request.data.get("system_prompt", "")

        t = threading.Thread(
            target=interpret_single_feature,
            args=(feature.id, model_name, prompt, k_positive, k_negative, temperature),
            daemon=True,
        )
        t.start()
        return Response({"status": "started", "feature_id": feature.id})

    @action(detail=True, methods=["post"])
    def interpret(self, request, pk=None):
        """Start batch interpretation pipeline for all features."""
        run = self.get_object()
        if run.status != RUN_COMPLETED:
            return Response(
                {"detail": "Run must be completed before interpretation."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        n_features = int(request.data.get("n_features", 20))
        ollama_model = request.data.get("ollama_model", "qwen2.5:14b")
        k_positive = int(request.data.get("k_positive", 6))
        k_negative = int(request.data.get("k_negative", 4))
        temperature = float(request.data.get("temperature", 0.2))
        system_prompt = request.data.get("system_prompt", "")

        t = threading.Thread(
            target=run_interpretation_pipeline,
            args=(run.id, n_features, ollama_model, system_prompt,
                  k_positive, k_negative, temperature),
            daemon=True,
        )
        t.start()
        return Response({"status": "started", "run_id": run.id})

    @action(detail=True, methods=["post"])
    def calculate_stats(self, request, pk=None):
        """Start statistics calculation pipeline."""
        run = self.get_object()
        t = threading.Thread(
            target=calculate_statistics_pipeline,
            args=(run.id,),
            daemon=True,
        )
        t.start()
        return Response({"status": "started", "run_id": run.id})

    @action(detail=True, methods=["get"])
    def families(self, request, pk=None):
        """List feature families for this SAE run."""
        run = self.get_object()
        qs = FeatureFamily.objects.filter(run=run)
        serializer = FeatureFamilySerializer(qs, many=True)
        return Response(serializer.data)


# ---------------------------------------------------------------------------
# Standalone endpoints
# ---------------------------------------------------------------------------

@api_view(["POST"])
def inference(request):
    """Text -> embedding -> SAE activations -> top features."""
    serializer = InferenceRequestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)

    run_id = serializer.validated_data["run_id"]
    text = serializer.validated_data["text"]

    run = get_object_or_404(SAERun, pk=run_id)

    # 1. Generate embedding
    embedder = get_embedder(run.dataset.model_name)
    embeddings = embedder.embed_texts([text])
    if not embeddings or not embeddings[0]:
        return Response(
            {"detail": "Failed to generate embedding."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    embedding = embeddings[0]
    emb_tensor = torch.tensor([embedding], dtype=torch.float32)

    # 2. Load SAE & run inference
    device = "cpu"
    model, mean, std = load_sae_model(run, device)
    if model is None:
        return Response(
            {"detail": "Failed to load SAE model."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    if mean is not None:
        emb_tensor = zscore_transform(emb_tensor, mean, std)

    with torch.no_grad():
        acts = model.encode(emb_tensor)[0]

    # 3. Format results
    threshold = 0.0001
    non_zero = torch.nonzero(acts > threshold).flatten()
    values = acts[non_zero].tolist()
    indices = non_zero.tolist()

    features_db = SAEFeature.objects.filter(
        run=run, feature_index__in=indices
    )
    features_map = {f.feature_index: f for f in features_db}

    results = []
    for idx, val in sorted(zip(indices, values), key=lambda x: -x[1]):
        feat = features_map.get(idx)
        results.append({
            "feature_index": idx,
            "activation": val,
            "label": feat.label if feat else "",
            "description": feat.description if feat else "",
        })

    return Response({
        "run_id": run.id,
        "text": text,
        "embedding_norm": float(torch.norm(torch.tensor(embedding))),
        "active_features": len(results),
        "features": results,
    })


@api_view(["GET"])
@permission_classes([AllowAny])
def system_status(request):
    """System status and version info."""
    # Check Ollama
    ollama_ok = False
    try:
        import requests as req
        from project.utils import get_setting
        resp = req.get(f"{get_setting('ollama_base_url')}/api/tags", timeout=5)
        ollama_ok = resp.status_code == 200
    except Exception:
        pass

    # Check OpenSearch
    opensearch_ok = False
    try:
        from search.client import is_available
        opensearch_ok = is_available()
    except Exception:
        pass

    return Response({
        "version": __version__,
        "status": "ok",
        "services": {
            "ollama": ollama_ok,
            "opensearch": opensearch_ok,
        },
    })


@api_view(["POST"])
def search_bm25(request):
    """BM25 keyword search."""
    serializer = SearchRequestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    d = serializer.validated_data

    try:
        from search.queries import search_documents_bm25
        results = search_documents_bm25(d["dataset_id"], d["query"], size=d.get("size", 10))
        return Response({"results": results})
    except Exception as e:
        return Response({"detail": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)


@api_view(["POST"])
def search_semantic(request):
    """Semantic (vector) search."""
    serializer = SearchRequestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    d = serializer.validated_data

    dataset = get_object_or_404(Dataset, pk=d["dataset_id"])
    embedder = get_embedder(dataset.model_name)
    embeddings = embedder.embed_texts([d["query"]])
    if not embeddings or not embeddings[0]:
        return Response(
            {"detail": "Failed to generate query embedding."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    try:
        from search.queries import search_similar_documents
        results = search_similar_documents(d["dataset_id"], embeddings[0], k=d.get("size", 10))
        return Response({"results": results})
    except Exception as e:
        return Response({"detail": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)


@api_view(["POST"])
def search_hybrid(request):
    """Hybrid search (BM25 + semantic)."""
    serializer = HybridSearchRequestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    d = serializer.validated_data

    dataset = get_object_or_404(Dataset, pk=d["dataset_id"])
    embedder = get_embedder(dataset.model_name)
    embeddings = embedder.embed_texts([d["query"]])
    if not embeddings or not embeddings[0]:
        return Response(
            {"detail": "Failed to generate query embedding."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    try:
        from search.queries import search_documents_hybrid
        results = search_documents_hybrid(
            d["dataset_id"], d["query"], embeddings[0],
            size=d.get("size", 10),
            bm25_weight=d["bm25_weight"],
            knn_weight=d["knn_weight"],
        )
        return Response({"results": results})
    except Exception as e:
        return Response({"detail": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
