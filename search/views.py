from django.shortcuts import get_object_or_404, render

from embeddings.embedders import get_embedder
from embeddings.models import Dataset
from sae.models import SAERun

from .queries import (
    search_documents_bm25,
    search_documents_hybrid,
    search_features,
    search_similar_documents,
)


def search_page(request):
    """Unified search page: hybrid document search + feature search."""
    datasets = Dataset.objects.all()
    runs = SAERun.objects.filter(status='completed')

    context = {
        'datasets': datasets,
        'runs': runs,
        'results': [],
        'feature_results': [],
        'query': '',
        'search_type': 'hybrid',
        'selected_dataset': None,
        'selected_run': None,
    }

    query = request.GET.get('q', '').strip()
    search_type = request.GET.get('type', 'hybrid')
    dataset_id = request.GET.get('dataset_id')
    run_id = request.GET.get('run_id')

    context['query'] = query
    context['search_type'] = search_type

    if not query:
        return render(request, 'search/search.html', context)

    # Document search
    if dataset_id:
        dataset = get_object_or_404(Dataset, pk=dataset_id)
        context['selected_dataset'] = dataset

        if search_type == 'bm25':
            context['results'] = search_documents_bm25(dataset.id, query, size=20)
        elif search_type == 'semantic':
            embedder = get_embedder(dataset.model_name)
            embeddings = embedder.embed_texts([query])
            if embeddings and embeddings[0]:
                context['results'] = search_similar_documents(
                    dataset.id, embeddings[0], k=20
                )
        elif search_type == 'hybrid':
            embedder = get_embedder(dataset.model_name)
            embeddings = embedder.embed_texts([query])
            if embeddings and embeddings[0]:
                context['results'] = search_documents_hybrid(
                    dataset.id, query, embeddings[0], size=20
                )
            else:
                # Fallback to BM25 if embedding fails
                context['results'] = search_documents_bm25(
                    dataset.id, query, size=20
                )

    # Feature search
    if run_id:
        run = get_object_or_404(SAERun, pk=run_id)
        context['selected_run'] = run
        context['feature_results'] = search_features(run.id, query, size=50)

    return render(request, 'search/search.html', context)
