# explorer/family_builder.py
import networkx as nx
import torch
from django.db import transaction
from .models import SAEFeature, FeatureFamily, SAERun

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
            
        # D. Estrai Famiglie (DFS / Componenti connesse dirette)
        # Una famiglia è definita da una radice locale (nodo con out-degree > 0 in questo albero)
        # Per semplicità, prendiamo ogni nodo che ha figli come "Genitore" di una famiglia locale.
        
        parents_in_this_iter = set()
        
        with transaction.atomic():
            for node in directed_tree.nodes():
                children = list(directed_tree.successors(node))
                if children:
                    # Abbiamo trovato una famiglia!
                    parents_in_this_iter.add(node)
                    
                    # Crea oggetto Famiglia
                    parent_obj = feat_map[node]
                    family = FeatureFamily.objects.create(
                        run=run,
                        parent_feature=parent_obj,
                        iteration=iteration,
                        size=len(children) + 1,
                        # Usiamo la label del genitore come nome provvisorio della famiglia
                        family_label=parent_obj.label or f"Family {node}" 
                    )
                    
                    # Aggiungi figli
                    child_objs = [feat_map[c] for c in children]
                    family.children_features.set(child_objs)
                    families_created += 1

        # E. Rimuovi i genitori per la prossima iterazione
        # "removing parent features after each iteration to re-form the MST"
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
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import io
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
        idx_map = {old: new for new, old in enumerate(active_indices)}
        
        if n_feats > 0:
            # --- Matrice S (Similarity) ---
            W_dec = model.decoder.weight.data.T # [total_feats, d_in]
            W_subset = W_dec[active_indices]
            W_norm = W_subset / (W_subset.norm(dim=1, keepdim=True) + 1e-8)
            S = torch.mm(W_norm, W_norm.T).numpy()
            
            # --- Matrice C (Co-occurrence) & D (Dense) ---
            # Scan veloce su un subset di documenti (max 2000 per velocità)
            C = torch.zeros((n_feats, n_feats))
            D = torch.zeros((n_feats, n_feats))
            
            docs = run.dataset.documents.filter(status='done')[:2000]
            embs = [d.embedding for d in docs if d.embedding]
            
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
                mask = np.eye(matrix.shape[0], dtype=bool)
                
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