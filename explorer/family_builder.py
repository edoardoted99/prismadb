# explorer/family_builder.py
import networkx as nx
import torch
from django.db import transaction

from .models import FeatureFamily, SAEFeature, SAERun


def build_feature_families(run_id, threshold=0.1, n_iterations=3):
    """
    Implementazione Sezione 4.2 del paper:
    1. Costruisci grafo da co-occorrenze normalizzate.
    2. Calcola Maximum Spanning Tree (MST).
    3. Orienta archi (Alta Densità -> Bassa Densità).
    4. Estrai famiglie (Nodi con out-degree > 0).
    5. Rimuovi nodi genitori e ripeti.
    """
    print(f"[Families] Building for Run #{run_id}...")
    run = SAERun.objects.get(pk=run_id)

    # 1. Carica tutte le feature e le loro co-occorrenze in memoria
    # (Assumiamo che calculate_statistics_pipeline sia stato eseguito e i dati siano in co_occurring_features)
    features = list(SAEFeature.objects.filter(run=run))
    feat_map = {f.feature_index: f for f in features}

    # Mappa densità: {feature_index: density}
    densities = {f.feature_index: (f.density or 0) for f in features}

    # Insieme dei nodi attivi (inizialmente tutti)
    active_nodes = set(densities.keys())

    families_created = 0

    # Puliamo famiglie vecchie per questa run
    FeatureFamily.objects.filter(run=run).delete()

    for iteration in range(1, n_iterations + 1):
        print(f"[Families] Iteration {iteration}/{n_iterations}. Active nodes: {len(active_nodes)}")

        if len(active_nodes) < 2:
            break

        # A. Costruisci Grafo (G) sui nodi attivi
        G = nx.Graph()
        G.add_nodes_from(active_nodes)

        for f in features:
            if f.feature_index not in active_nodes:
                continue

            if f.co_occurring_features:
                for co in f.co_occurring_features:
                    target_idx = co['index']
                    weight = co['score'] # Probabilità condizionata (C_norm)

                    if target_idx in active_nodes and weight > threshold:
                        # MST minimizza, quindi usiamo peso negativo o parametro 'maximum'
                        G.add_edge(f.feature_index, target_idx, weight=weight)

        if G.number_of_edges() == 0:
            print("[Families] No edges found with current threshold.")
            break

        # B. Calcola Maximum Spanning Tree
        mst = nx.maximum_spanning_tree(G, weight='weight')

        # C. Orienta Archi (Parent -> Child basato su Densità)
        directed_tree = nx.DiGraph()
        for u, v in mst.edges():
            # Il paper dice: edges pointing from higher-density to lower-density
            if densities.get(u, 0) >= densities.get(v, 0):
                parent, child = u, v
            else:
                parent, child = v, u

            directed_tree.add_edge(parent, child)

        # D. Extract families via DFS from root nodes (Paper §4.2)
        # Root nodes = no incoming edges, with at least one outgoing edge
        roots = [n for n in directed_tree.nodes()
                 if directed_tree.in_degree(n) == 0
                 and directed_tree.out_degree(n) > 0]
        parents_in_this_iter = set()

        with transaction.atomic():
            for root in roots:
                # DFS to collect all descendants
                descendants = list(nx.dfs_preorder_nodes(directed_tree, root))
                descendants.remove(root)

                if not descendants:
                    continue

                parents_in_this_iter.add(root)

                parent_obj = feat_map[root]
                family = FeatureFamily.objects.create(
                    run=run,
                    parent_feature=parent_obj,
                    iteration=iteration,
                    size=len(descendants) + 1,
                    family_label=parent_obj.label or f"Family {root}"
                )

                child_objs = [feat_map[c] for c in descendants]
                family.children_features.set(child_objs)
                families_created += 1

        # De-duplicate families with Jaccard overlap > 0.6 (Paper §4.2)
        iter_families = list(FeatureFamily.objects.filter(run=run, iteration=iteration))
        to_delete = set()
        for i, f1 in enumerate(iter_families):
            if f1.id in to_delete:
                continue
            set1 = set(f1.children_features.values_list('feature_index', flat=True))
            set1.add(f1.parent_feature.feature_index)
            for f2 in iter_families[i+1:]:
                if f2.id in to_delete:
                    continue
                set2 = set(f2.children_features.values_list('feature_index', flat=True))
                set2.add(f2.parent_feature.feature_index)
                jaccard = len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0
                if jaccard > 0.6:
                    # Keep the larger family
                    if f1.size >= f2.size:
                        to_delete.add(f2.id)
                    else:
                        to_delete.add(f1.id)
                        break
        if to_delete:
            deleted_count = len(to_delete)
            FeatureFamily.objects.filter(id__in=to_delete).delete()
            families_created -= deleted_count
            print(f"[Families] De-duplicated {deleted_count} families (Jaccard > 0.6)")

        # E. Remove parent features for next iteration (Paper §4.2)
        active_nodes -= parents_in_this_iter

        if not parents_in_this_iter:
            break

    print(f"[Families] Done. Created {families_created} families.")

    # ====================================================
    # F. Generazione Matrici (S, C, D) per Visualizzazione
    # ====================================================
    print("[Families] Generating Matrix Heatmaps...")
    try:
        import matplotlib
        matplotlib.use('Agg') # Backend non interattivo
        import io

        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        from django.core.files.base import ContentFile

        from sae.modules import SAE, SAEConfig, zscore_transform

        # 1. Carica Modello e Dati
        device = "cpu" # Usiamo CPU per sicurezza su plot
        if not run.weights_file:
            print("[Families] No weights file, skipping matrices.")
            return

        ckpt = torch.load(run.weights_file.path, map_location=device)
        cfg_dict = ckpt['config']; cfg_dict['device'] = device
        model = SAE(SAEConfig(**cfg_dict)).to(device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        mean = ckpt['zscore_mean'].to(device)
        std = ckpt['zscore_std'].to(device)

        # Recupera feature attive (quelle nel DB)
        active_indices = sorted(list(feat_map.keys()))
        n_feats = len(active_indices)
        _ = {old: new for new, old in enumerate(active_indices)}

        if n_feats > 0:
            # --- Matrice S (Similarity) ---
            W_dec = model.decoder.weight.data.T # [total_feats, d_in]
            W_subset = W_dec[active_indices]
            W_norm = W_subset / (W_subset.norm(dim=1, keepdim=True) + 1e-8)
            S = torch.mm(W_norm, W_norm.T).numpy()

            # --- Matrice C (Co-occurrence) & D (Dense) ---
            C = torch.zeros((n_feats, n_feats))
            D = torch.zeros((n_feats, n_feats))

            from search.bulk_ops import scroll_documents_in_batches
            embs = []
            for batch_data in scroll_documents_in_batches(run.dataset_id, batch_size=500,
                                                           fields=['embedding']):
                for d in batch_data:
                    if d.get('embedding') is not None:
                        embs.append(d['embedding'])
                    if len(embs) >= 2000:
                        break
                if len(embs) >= 2000:
                    break

            if embs:
                X = torch.tensor(embs, dtype=torch.float32).to(device)
                X = zscore_transform(X, mean, std)
                with torch.no_grad():
                    _, _, h_sparse = model(X) # [B, total_feats]

                # Filtra solo colonne attive
                h_subset = h_sparse[:, active_indices] # [B, n_feats]

                # C = A^T A (Binarizzato)
                A = (h_subset > 0).float()
                C = torch.mm(A.T, A)

                # Normalizzazione C (Paper: C_ij / (f_i + epsilon))
                epsilon = 1e-5
                freqs = A.sum(dim=0) # f_i
                # C_norm[i, j] = C[i, j] / (f[i] + epsilon)
                # Broadcasting: C (NxN) / freqs (Nx1)
                C_norm = C / (freqs.unsqueeze(1) + epsilon)
                C_np = C_norm.numpy()

                # D = V^T V (Valori Reali)
                D = torch.mm(h_subset.T, h_subset)
                # Normalizzazione D (Simile a cosine sim delle attivazioni)
                h_norm = h_subset.norm(dim=0, keepdim=True) + 1e-8
                D_norm = D / (h_norm.T @ h_norm)
                D_np = D_norm.numpy()
            else:
                C_np = np.zeros((n_feats, n_feats))
                D_np = np.zeros((n_feats, n_feats))

            # --- Plotting Function ---
            def save_heatmap(matrix, title, filename_prefix):
                plt.figure(figsize=(12, 10), facecolor='black') # Increased size for annotations
                ax = plt.gca()
                ax.set_facecolor('black')

                # Maschera diagonale per visualizzazione migliore (opzionale, ma utile)
                _mask = np.eye(matrix.shape[0], dtype=bool)

                hm = sns.heatmap(
                    matrix,
                    cmap='inferno',
                    annot=False, # Hide values
                    xticklabels=False, # Hide feature IDs on axis
                    yticklabels=False,
                    square=True,
                    cbar_kws={"shrink": .8}
                )

                # Style Colorbar
                cbar = hm.collections[0].colorbar
                cbar.ax.tick_params(labelsize=10, colors='white')
                cbar.outline.set_edgecolor('white')

                plt.title(f"{title} (k={model.k})", color='white', fontsize=16, pad=20)
                plt.xlabel("Feature Index", color='white', fontsize=12)
                plt.ylabel("Feature Index", color='white', fontsize=12)
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', facecolor='black', dpi=100)
                plt.close()
                return ContentFile(buf.getvalue(), name=f"{filename_prefix}_{run.id}.png")

            # Salva Immagini
            print("[Families] Saving S Matrix...")
            run.matrix_s_heatmap = save_heatmap(S, "Decoder Weights Similarity (S)", "matrix_s")

            print("[Families] Saving C Matrix...")
            run.matrix_c_heatmap = save_heatmap(C_np, "Co-occurrence Matrix (C)", "matrix_c")

            print("[Families] Saving D Matrix...")
            run.matrix_d_heatmap = save_heatmap(D_np, "Activation Similarity (D)", "matrix_d")

            run.save()
            print("[Families] Matrices saved successfully.")

    except Exception as e:
        print(f"[Families] Error generating matrices: {e}")
        import traceback
        traceback.print_exc()
