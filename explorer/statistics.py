import logging
import threading
import time
from collections import defaultdict

import numpy as np
import torch

from project.utils import get_device
from sae.models import SAERun
from sae.modules import SAE, SAEConfig, zscore_transform

from .models import SAEFeature
from .task_status import TASK_PROGRESS

logger = logging.getLogger(__name__)

N_BINS = 100

def calculate_statistics_pipeline(run_id):
    """
    Calcola correlazioni (pesi), co-occorrenze (dati) e istogrammi per le feature.
    """
    logger.info(f"[Stats] Starting analysis for Run #{run_id}")

    tid = threading.get_ident()
    TASK_PROGRESS[tid] = {
        'progress': 0,
        'message': 'Initializing statistics calculation...',
        'start_time': time.time(),
        'run_id': run_id
    }

    try:
        run = SAERun.objects.get(pk=run_id)
        device = get_device()

        if not run.weights_file:
            logger.error("[Stats] No weights file.")
            TASK_PROGRESS[tid].update({'progress': 100, 'message': 'Error: No weights file.'})
            return

        # 1. Carica Modello
        TASK_PROGRESS[tid].update({'progress': 5, 'message': 'Loading SAE model...'})
        ckpt = torch.load(run.weights_file.path, map_location=device)

        # Forza il device corretto nel config caricato
        cfg_dict = ckpt['config']
        cfg_dict['device'] = device

        cfg = SAEConfig(**cfg_dict)
        model = SAE(cfg).to(device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        # Pesi del Decoder per correlazione semantica
        W_dec = model.decoder.weight.data.T  # [n_features, d_in]
        W_norm = W_dec / (W_dec.norm(dim=1, keepdim=True) + 1e-8)

        mean = ckpt['zscore_mean'].to(device)
        std = ckpt['zscore_std'].to(device)

        # Recuperiamo le feature già create
        existing_features = list(SAEFeature.objects.filter(run=run).values_list('feature_index', flat=True))
        if not existing_features:
            logger.warning("[Stats] No features found in DB. Run interpreter first.")
            TASK_PROGRESS[tid].update({'progress': 100, 'message': 'Error: No features found.'})
            return

        logger.info(f"[Stats] Analyzing {len(existing_features)} features...")

        # ====================================================
        # A. Correlazione Pesi (Cosine Similarity)
        # ====================================================
        TASK_PROGRESS[tid].update({'progress': 10, 'message': 'Calculating weight correlations...'})
        target_indices = torch.tensor(existing_features, device=device)
        sim_matrix = torch.mm(W_norm[target_indices], W_norm.T)

        correlations_update = {}

        vals, idxs = torch.topk(sim_matrix, k=6, dim=1)
        vals = vals.cpu().numpy()
        idxs = idxs.cpu().numpy()

        for i, f_idx in enumerate(existing_features):
            corrs = []
            for rank in range(1, 6):
                other_idx = int(idxs[i, rank])
                score = float(vals[i, rank])
                label = f"Feature {other_idx}"
                corrs.append({'index': other_idx, 'score': round(score, 4), 'label': label})
            correlations_update[f_idx] = corrs

        # ====================================================
        # B. Co-Occorrenze e Istogrammi (Scan Dati)
        # ====================================================
        logger.info("[Stats] Scanning dataset for co-occurrences...")
        TASK_PROGRESS[tid].update({'progress': 20, 'message': 'Scanning dataset...'})

        co_occurrences = {f: defaultdict(int) for f in existing_features}
        activation_values = {f: [] for f in existing_features}
        feature_counts = defaultdict(int)

        batch_size = 256
        target_set = set(existing_features)
        processed = 0

        # Try ChromaDB scroll first, fallback to SQLite
        use_chromadb = False
        try:
            from search.client import is_available
            if is_available():
                from search.bulk_ops import count_documents, scroll_documents_in_batches
                total_docs = count_documents(run.dataset_id)
                use_chromadb = True
                logger.info("[Stats] Using ChromaDB for dataset scan.")
        except Exception:
            use_chromadb = False

        if not use_chromadb:
            doc_qs = run.dataset.documents.filter(status='done').order_by('id')
            total_docs = doc_qs.count()
            logger.info("[Stats] Using SQLite for dataset scan.")

        def _get_batch_iterator():
            """Yields lists of embeddings per batch."""
            if use_chromadb:
                for batch_data in scroll_documents_in_batches(run.dataset_id, batch_size=batch_size,
                                                               fields=['embedding']):
                    emb_list = [d['embedding'] for d in batch_data if d.get('embedding')]
                    yield emb_list, len(batch_data)
            else:
                offset = 0
                while offset < total_docs:
                    batch_docs = list(doc_qs[offset:offset+batch_size])
                    if not batch_docs: break
                    emb_list = [d.embedding for d in batch_docs if d.embedding]
                    yield emb_list, len(batch_docs)
                    offset += batch_size

        for emb_list, batch_len in _get_batch_iterator():
            if not emb_list:
                processed += batch_len
                continue

            X = torch.tensor(emb_list, dtype=torch.float32).to(device)
            X = zscore_transform(X, mean, std)

            with torch.no_grad():
                _, _, h_sparse = model(X)

            h_cpu = h_sparse.cpu()
            indices = torch.nonzero(h_cpu)
            values = h_cpu[indices[:,0], indices[:,1]]

            # Raggruppa per documento
            docs_map = defaultdict(list)

            # 1. Accumula valori
            rows_np = indices[:, 0].numpy()
            feats_np = indices[:, 1].numpy()
            vals_np = values.numpy()

            for r, f, v in zip(rows_np, feats_np, vals_np):
                fid = int(f)
                if fid in target_set:
                    activation_values[fid].append(float(v))
                    feature_counts[fid] += 1
                    docs_map[int(r)].append(fid)

            # 2. Calcola Co-occorrenze
            for row, feats in docs_map.items():
                for f1 in feats:
                    if f1 in target_set:
                        for f2 in feats:
                            if f1 != f2:
                                co_occurrences[f1][f2] += 1

            processed += batch_len

            # Progress 20% -> 80%
            pct = 20 + int((processed / total_docs) * 60) if total_docs > 0 else 80
            TASK_PROGRESS[tid].update({'progress': pct, 'message': f'Scanning dataset ({processed}/{total_docs})...'})

            if processed % 1000 == 0:
                logger.info(f"[Stats] Processed {processed}/{total_docs} docs...")

        # ====================================================
        # C. Finalizzazione e Salvataggio
        # ====================================================
        logger.info("[Stats] Saving results to DB...")
        TASK_PROGRESS[tid].update({'progress': 85, 'message': 'Saving statistics to DB...'})

        features_to_update = []
        # Carichiamo in un colpo solo per evitare N query
        db_features = list(SAEFeature.objects.filter(run=run, feature_index__in=existing_features))

        for feat in db_features:
            fid = feat.feature_index

            # 1. Aggiorna Correlazioni
            feat.correlated_features = correlations_update.get(fid, [])

            # 2. Aggiorna Co-occorrenze
            co_list = co_occurrences.get(fid, {})
            sorted_co = sorted(co_list.items(), key=lambda x: x[1], reverse=True)[:6]

            feat_co_data = []
            for other_fid, count in sorted_co:
                feat_co_data.append({
                    'index': other_fid,
                    'label': f"Feature {other_fid}",
                    'count': count,
                    'score': round(count / (feature_counts[fid] + 1e-5), 2)
                })
            feat.co_occurring_features = feat_co_data

            # 3. Aggiorna Istogramma e Densità
            acts = activation_values.get(fid, [])
            if acts:
                counts, bin_edges = np.histogram(acts, bins=N_BINS, density=False)
                feat.activation_histogram = {
                    'counts': counts.tolist(),
                    'bins': [round(b, 3) for b in bin_edges.tolist()]
                }
                # Ricalcola density precisa
                feat.density = len(acts) / total_docs
                # Ricalcola max reale
                feat.max_activation = max(acts)

                # Calcola Mean e Variance (solo su attivazioni > 0)
                if len(acts) > 0:
                    feat.mean_activation = float(np.mean(acts))
                    feat.variance_activation = float(np.var(acts))
                else:
                    feat.mean_activation = 0.0
                    feat.variance_activation = 0.0

            features_to_update.append(feat)

        SAEFeature.objects.bulk_update(
            features_to_update,
            ['correlated_features', 'co_occurring_features', 'activation_histogram', 'density', 'max_activation', 'mean_activation', 'variance_activation']
        )

        logger.info("[Stats] Statistics calculation completed successfully.")
        TASK_PROGRESS[tid].update({'progress': 100, 'message': 'Done.'})

    except Exception as e:
        logger.error(f"[Stats] Critical Error: {e}", exc_info=True)
        TASK_PROGRESS[tid].update({'progress': 100, 'message': f'Error: {e}'})
