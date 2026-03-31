# sae/trainer.py
import io
import time

import matplotlib
import matplotlib.pyplot as plt
import torch
from django.core.files.base import ContentFile
from torch.utils.data import DataLoader, TensorDataset

from .models import SAERun
from .modules import (
    SAE,
    SAEConfig,
    compute_zscore_stats,
    get_device,
    sae_loss_func,
    zscore_transform,
)

# Backend non interattivo per il server
matplotlib.use('Agg')


def generate_heatmap(run, model, X_data, device):
    """
    Genera la heatmap (Documenti/Chunks x Neuroni) con tema SCURO (Black).
    """
    try:
        # --- 1. Preparazione Dati (Invariata) ---
        MAX_SAMPLES = 10000
        n_samples = min(MAX_SAMPLES, len(X_data))

        # Prendi i primi n_samples (o randomizza se preferisci scommentando la riga sotto)
        # idx = torch.randperm(len(X_data))[:n_samples]
        # X_sample = X_data[idx].to(device)
        X_sample = X_data[:n_samples].to(device)

        model.eval()
        with torch.no_grad():
            _, h, _ = model(X_sample)
            h = h.cpu()

        # Ordinamento per attività totale dei neuroni
        activation_score = h.sum(dim=0)
        indices = torch.argsort(activation_score, descending=True)

        # Prendiamo i top M neuroni
        TOP_M = min(500, h.shape[1])
        top_indices = indices[:TOP_M]

        h_sorted = h[:, top_indices].numpy()

        # Normalizzazione per riga (per visualizzare meglio i pattern relativi)
        mn = h_sorted.min(axis=1, keepdims=True)
        mx = h_sorted.max(axis=1, keepdims=True)
        h_norm = (h_sorted - mn) / (mx - mn + 1e-9)

        # --- 2. Plotting con Stile Dark ---

        # Usiamo il contesto 'dark_background' per invertire colori assi e testo automaticamente
        with plt.style.context('dark_background'):

            fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

            # Forziamo lo sfondo a nero puro (a volte il default è grigio scuro)
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')

            # Disegno la heatmap
            # 'inferno' o 'magma' sono ottime su sfondo nero perché partono dal nero
            im = ax.imshow(h_norm, aspect="auto", interpolation="nearest", cmap="inferno")

            # Titoli e label (saranno bianchi grazie allo stile)
            ax.set_title(f"Sparsity Heatmap: Chunk x Top-{TOP_M} Neurons\n(Run #{run.id})", color='white', fontsize=12, fontweight='bold')
            ax.set_xlabel("Active Neurons", color='white')
            ax.set_ylabel("Text Chunks", color='white')

            # Colorbar personalizzata
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Activation (Normalized)", color='white')
            cbar.ax.yaxis.set_tick_params(color='white') # Tacchette bianche
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white') # Numeri bianchi
            cbar.outline.set_edgecolor('white') # Bordo bianco

            # --- 3. Salvataggio ---
            buffer = io.BytesIO()
            # facecolor='black' assicura che anche i margini salvati siano neri
            fig.savefig(buffer, format='png', dpi=150, facecolor='black')
            plt.close(fig)
            buffer.seek(0)

            run.sparsity_heatmap.save(f"heatmap_run_{run.id}.png", ContentFile(buffer.read()), save=False)

    except Exception as e:
        print(f"[Run #{run.id}] Heatmap Error: {e}")


def train_sae_run(run_id: int):
    """
    Esegue il training gestendo il 'Flattening' dei chunk.
    """
    try:
        run = SAERun.objects.get(pk=run_id)
        run.status = 'running'
        run.save()

        print(f"[Run #{run.id}] Loading embeddings...")

        # 1. Recupero dati - try OpenSearch first, fallback to SQLite
        docs_vectors = None
        try:
            from search.client import is_available
            if is_available():
                from search.bulk_ops import scroll_all_embeddings
                docs_vectors = [emb for _, emb in scroll_all_embeddings(run.dataset_id)]
                print(f"[Run #{run.id}] Loaded {len(docs_vectors)} embeddings from OpenSearch")
        except Exception as e:
            print(f"[Run #{run.id}] OpenSearch unavailable ({e}), falling back to SQLite")

        if not docs_vectors:
            docs_vectors = list(run.dataset.documents.filter(status='done').values_list('embedding', flat=True))

        # 2. Flattening (Schiacciamento)
        flat_embeddings = []
        for doc_vecs in docs_vectors:
            if not doc_vecs: continue

            # Gestione robusta: controlla se è lista di liste o lista di float (vecchio formato)
            if isinstance(doc_vecs, list) and len(doc_vecs) > 0:
                if isinstance(doc_vecs[0], list):
                    # Caso corretto: Documento -> [Chunk1, Chunk2]
                    flat_embeddings.extend(doc_vecs)
                elif isinstance(doc_vecs[0], (float, int)):
                    # Caso legacy: Documento -> Vettore singolo
                    flat_embeddings.append(doc_vecs)

        if not flat_embeddings:
            raise ValueError("No valid embeddings found in dataset.")

        # 3. Creazione Tensore Unico
        # X shape: [Totale_Chunks, Dimensione_Embedding]
        X = torch.tensor(flat_embeddings, dtype=torch.float32)
        print(f"[Run #{run.id}] Flattened dataset shape: {X.shape}")

        # 4. Z-Score Normalization
        mean, std = compute_zscore_stats(X)
        X_std = zscore_transform(X, mean, std)

        # 5. Setup Modello
        device = get_device()
        d_latent = run.input_dim * run.expansion_factor

        cfg = SAEConfig(
            d_in=run.input_dim,
            d_latent=d_latent,
            k=run.k_sparsity,
            alpha_aux=run.alpha_aux,
            lr=run.learning_rate,
            device=device
        )

        model = SAE(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

        dataset = TensorDataset(X_std)
        loader = DataLoader(dataset, batch_size=run.batch_size, shuffle=True)

        # 6. Training Loop
        print(f"[Run #{run.id}] Training started...")
        model.train()
        log_history = []

        for epoch in range(1, run.epochs + 1):
            model.start_epoch()
            epoch_loss = 0.0
            steps = 0

            for (xb,) in loader:
                xb = xb.to(device)

                out = sae_loss_func(model, xb, alpha_aux=cfg.alpha_aux)

                optimizer.zero_grad()
                out.total.backward()
                optimizer.step()

                epoch_loss += out.total.item()
                steps += 1

            avg_loss = epoch_loss / steps if steps > 0 else 0

            print(f"[Run #{run.id}] Epoch {epoch} - Loss: {avg_loss:.4f}")
            log_history.append({"epoch": epoch, "loss": round(avg_loss, 5), "timestamp": time.time()})
            run.training_log = log_history
            run.save(update_fields=['training_log'])

        # 7. Analisi e Salvataggio
        print(f"[Run #{run.id}] Generating Analysis Plots...")
        generate_heatmap(run, model, X_std, device)

        checkpoint = {
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
            "zscore_mean": mean,
            "zscore_std": std
        }

        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        run.weights_file.save(f"sae_run_{run_id}_final.pt", ContentFile(buffer.read()), save=False)

        run.final_loss = log_history[-1]['loss'] if log_history else 0
        run.status = 'completed'
        run.save()
        print(f"[Run #{run.id}] Completed successfully.")

    except Exception as e:
        print(f"[Run #{run.id}] Failed: {e}")
        # Ricarica l'oggetto per evitare conflitti di scrittura
        run = SAERun.objects.get(pk=run_id)
        run.status = 'failed'
        run.error_message = str(e)
        run.save()
