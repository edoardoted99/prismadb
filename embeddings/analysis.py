import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from .models import Dataset, Document

def perform_dataset_analysis(dataset_id):
    """
    Esegue l'analisi completa su un dataset:
    1. Token counts
    2. Embedding norms & outliers
    3. PCA / Clustering
    4. Global stats
    """
    dataset = Dataset.objects.get(pk=dataset_id)
    docs = dataset.documents.filter(status="done").exclude(embedding__isnull=True)
    
    if not docs.exists():
        return

    # 1. Token Counts (approssimato split spazi se non abbiamo tokenizer specifico)
    # Se volessimo essere precisi useremmo tiktoken o il tokenizer del modello, 
    # ma per ora split() è una buona proxy veloce.
    for doc in docs:
        doc.token_count = len(doc.text.split())
    
    # Bulk update token counts
    Document.objects.bulk_update(docs, ["token_count"])

    # Carichiamo embeddings in numpy
    embeddings = np.array([d.embedding for d in docs])
    
    # 2. Norms & Outliers
    norms = np.linalg.norm(embeddings, axis=1)
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    
    # Definiamo outlier se norm dista più di 2 deviazioni standard
    outlier_indices = np.where(np.abs(norms - mean_norm) > 2 * std_norm)[0]
    
    # 3. PCA & Clustering
    # PCA a 2 componenti
    if len(docs) > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
    else:
        coords = np.zeros((len(docs), 2))

    # Clustering (K-Means)
    # Scegliamo K in base al numero di documenti, max 5 cluster per ora
    n_clusters = min(5, len(docs))
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
    else:
        labels = np.zeros(len(docs), dtype=int)

    # Aggiorniamo i documenti
    docs_to_update = []
    for i, doc in enumerate(docs):
        doc.embedding_norm = float(norms[i])
        doc.is_outlier = i in outlier_indices
        doc.pca_x = float(coords[i, 0])
        doc.pca_y = float(coords[i, 1])
        doc.cluster_label = int(labels[i])
        docs_to_update.append(doc)
    
    Document.objects.bulk_update(
        docs_to_update, 
        ["embedding_norm", "is_outlier", "pca_x", "pca_y", "cluster_label"]
    )

    # 4. Global Stats
    # Calcoliamo similarità media (su un sample se troppi documenti)
    if len(docs) > 1000:
        sample_indices = np.random.choice(len(embeddings), 1000, replace=False)
        sample_emb = embeddings[sample_indices]
    else:
        sample_emb = embeddings

    sim_matrix = cosine_similarity(sample_emb)
    # Prendiamo solo triangolo superiore escluso diagonale
    upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    avg_similarity = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0

    dataset.analysis_metadata = {
        "avg_similarity": avg_similarity,
        "mean_norm": float(mean_norm),
        "std_norm": float(std_norm),
        "n_clusters": n_clusters,
        "n_outliers": len(outlier_indices),
        "avg_tokens": float(np.mean([d.token_count for d in docs])),
    }
    dataset.save()
