#!/usr/bin/env python
"""prismadb CLI — manage datasets, SAE training, and search from the terminal."""
import os
import sys

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "project.settings")

import click


def _setup_django():
    import django
    django.setup()


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="prismadb")
def cli():
    """prismadb — Sparse Autoencoder explorer for LLM embeddings."""
    pass


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@cli.command()
def init():
    """Initialize database and ChromaDB."""
    _setup_django()
    from django.conf import settings
    from django.core.management import call_command

    click.echo(f"Data directory: {settings.PRISMADB_HOME}")
    click.echo("Running migrations...")
    call_command("migrate", verbosity=0)
    click.echo("Migrations complete.")

    try:
        from search.client import is_available
        if is_available():
            click.echo("ChromaDB initialized.")
        else:
            click.echo("ChromaDB not available (install prismadb[search]).")
    except Exception:
        click.echo("ChromaDB not available (install prismadb[search]).")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--model", "-m", default="nomic-embed-text", help="Ollama embedding model name.")
@click.option("--name", "-n", default=None, help="Dataset name (defaults to filename).")
@click.option("--description", "-d", default="", help="Dataset description.")
@click.option("--embed/--no-embed", default=True, help="Generate embeddings after ingestion.")
@click.option("--batch-size", default=32, help="Embedding batch size.")
def ingest(file, model, name, description, embed, batch_size):
    """Ingest a JSON dataset and optionally generate embeddings.

    FILE must be a JSON file: [{"id": "1", "text": "..."}, ...]
    """
    _setup_django()
    from embeddings.services import generate_embeddings_for_dataset, ingest_json_and_create_dataset

    if name is None:
        name = os.path.splitext(os.path.basename(file))[0]

    click.echo(f"Ingesting {file} as '{name}' with model {model}...")
    with open(file, "rb") as f:
        dataset = ingest_json_and_create_dataset(f, name, description, model)

    n_docs = dataset.documents.count()
    click.echo(f"Created dataset #{dataset.id} with {n_docs} documents.")

    if embed:
        click.echo("Generating embeddings...")
        try:
            from tqdm import tqdm
            pbar = tqdm(total=n_docs, unit="doc", desc="Embedding")

            def _progress(processed, total):
                pbar.update(processed - pbar.n)

            generate_embeddings_for_dataset(dataset.id, batch_size=batch_size, progress_callback=_progress)
            pbar.close()
        except ImportError:
            generate_embeddings_for_dataset(dataset.id, batch_size=batch_size)

        done = dataset.documents.filter(status="done").count()
        click.echo(f"Embeddings complete: {done}/{n_docs} documents.")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--dataset", "-d", required=True, type=int, help="Dataset ID.")
@click.option("--expansion", default=4, help="Latent expansion factor.")
@click.option("--top-k", default=32, help="Top-K sparsity.")
@click.option("--epochs", default=20, help="Training epochs.")
@click.option("--lr", default=1e-4, type=float, help="Learning rate.")
@click.option("--batch-size", default=512, help="Training batch size.")
@click.option("--alpha-aux", default=0.03, type=float, help="Auxiliary loss coefficient.")
def train(dataset, expansion, top_k, epochs, lr, batch_size, alpha_aux):
    """Train a Sparse Autoencoder on a dataset's embeddings."""
    _setup_django()
    from embeddings.models import Dataset
    from project.constants import DOC_DONE, RUN_QUEUED
    from sae.models import SAERun
    from sae.trainer import train_sae_run

    ds = Dataset.objects.get(pk=dataset)
    first_doc = ds.documents.filter(status=DOC_DONE).first()
    if not first_doc or not first_doc.embedding:
        click.echo("Error: no embedded documents found. Run 'prismadb ingest' first.", err=True)
        sys.exit(1)

    input_dim = len(first_doc.embedding)

    run = SAERun.objects.create(
        dataset=ds,
        input_dim=input_dim,
        expansion_factor=expansion,
        k_sparsity=top_k,
        epochs=epochs,
        learning_rate=lr,
        batch_size=batch_size,
        alpha_aux=alpha_aux,
        status=RUN_QUEUED,
    )
    click.echo(f"Created SAE run #{run.id} (input_dim={input_dim}, latent={input_dim * expansion}, top-k={top_k})")
    click.echo("Training...")

    train_sae_run(run.id)

    run.refresh_from_db()
    click.echo(f"Training complete. Final loss: {run.final_loss:.6f}" if run.final_loss else "Training complete.")


# ---------------------------------------------------------------------------
# interpret
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--run", "-r", required=True, type=int, help="SAE run ID.")
@click.option("--model", "-m", default="qwen2.5:14b", help="Ollama LLM for interpretation.")
@click.option("--features", "-n", default=50, help="Number of top features to interpret.")
@click.option("--k-pos", default=5, help="Positive examples per feature.")
@click.option("--k-neg", default=5, help="Negative examples per feature.")
@click.option("--temperature", default=0.2, type=float, help="LLM temperature.")
def interpret(run, model, features, k_pos, k_neg, temperature):
    """Interpret SAE features using an LLM."""
    _setup_django()
    from explorer.interpreter import run_interpretation_pipeline

    click.echo(f"Interpreting top {features} features of run #{run} with {model}...")
    run_interpretation_pipeline(
        run, features_to_analyze=features, ollama_model=model,
        k_pos=k_pos, k_neg=k_neg, temp=temperature,
    )
    click.echo("Interpretation complete.")


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--run", "-r", required=True, type=int, help="SAE run ID.")
def stats(run):
    """Calculate feature statistics (correlations, co-occurrences, histograms)."""
    _setup_django()
    from explorer.statistics import calculate_statistics_pipeline

    click.echo(f"Calculating statistics for run #{run}...")
    calculate_statistics_pipeline(run)
    click.echo("Statistics complete.")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("query")
@click.option("--dataset", "-d", required=True, type=int, help="Dataset ID.")
@click.option("--mode", type=click.Choice(["bm25", "semantic", "hybrid"]), default="hybrid", help="Search mode.")
@click.option("--top-k", "-k", default=10, help="Number of results.")
def search(query, dataset, mode, top_k):
    """Search documents in a dataset."""
    _setup_django()
    from search.client import is_available

    if not is_available():
        click.echo("Error: ChromaDB is not available. Install prismadb[search].", err=True)
        sys.exit(1)

    from embeddings.embedders import get_embedder
    from embeddings.models import Dataset
    from search.queries import (
        search_documents_bm25,
        search_documents_hybrid,
        search_similar_documents,
    )

    results = []

    if mode == "bm25":
        results = search_documents_bm25(dataset, query, size=top_k)

    elif mode == "semantic":
        ds = Dataset.objects.get(pk=dataset)
        embedder = get_embedder(ds.model_name)
        embs = embedder.embed_texts([query])
        if embs and embs[0]:
            results = search_similar_documents(dataset, embs[0], k=top_k)

    elif mode == "hybrid":
        ds = Dataset.objects.get(pk=dataset)
        embedder = get_embedder(ds.model_name)
        embs = embedder.embed_texts([query])
        if embs and embs[0]:
            results = search_documents_hybrid(dataset, query, embs[0], size=top_k)

    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        text = r.get("text", "")[:200]
        click.echo(f"\n[{i}] score={score:.4f}")
        click.echo(f"    {text}...")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@cli.command("list")
@click.argument("resource", type=click.Choice(["datasets", "runs", "features"]))
@click.option("--run", "-r", type=int, default=None, help="SAE run ID (for 'features' resource).")
def list_resources(resource, run):
    """List datasets, SAE runs, or features."""
    _setup_django()

    if resource == "datasets":
        from embeddings.models import Dataset
        for ds in Dataset.objects.all().order_by("-created_at"):
            n_docs = ds.documents.count()
            n_done = ds.documents.filter(status="done").count()
            click.echo(f"  #{ds.id}  {ds.name:<30s}  model={ds.model_name}  docs={n_done}/{n_docs}")

    elif resource == "runs":
        from sae.models import SAERun
        for r in SAERun.objects.all().order_by("-created_at"):
            loss = f"{r.final_loss:.6f}" if r.final_loss else "—"
            click.echo(
                f"  #{r.id}  dataset={r.dataset.name:<20s}  "
                f"dim={r.input_dim}x{r.expansion_factor}  k={r.k_sparsity}  "
                f"status={r.status}  loss={loss}"
            )

    elif resource == "features":
        if run is None:
            click.echo("Error: --run is required for 'features'.", err=True)
            sys.exit(1)
        from explorer.models import SAEFeature
        qs = SAEFeature.objects.filter(run_id=run).order_by("-max_activation")
        for f in qs[:50]:
            label = f.label or "(unlabeled)"
            click.echo(f"  #{f.feature_index:<5d}  {label:<40s}  density={f.density or 0:.4f}  max_act={f.max_activation or 0:.4f}")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--port", "-p", default=8000, help="Port to listen on.")
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind to.")
def serve(port, host):
    """Start the web UI (Django development server)."""
    _setup_django()
    from django.core.management import call_command
    call_command("runserver", f"{host}:{port}")


# ---------------------------------------------------------------------------
# manage (escape hatch)
# ---------------------------------------------------------------------------

@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
def manage(ctx):
    """Run any Django management command (escape hatch).

    Example: prismadb manage createsuperuser
    """
    _setup_django()
    from django.core.management import execute_from_command_line
    execute_from_command_line(["manage"] + ctx.args)


def main():
    """Legacy entry point (calls the Click group)."""
    cli()


if __name__ == "__main__":
    main()
