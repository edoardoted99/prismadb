import threading  # <--- Assicurati che questa riga ci sia!

from django.contrib import messages
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.shortcuts import get_object_or_404, redirect, render

from .forms import UploadDatasetForm
from .models import Dataset, Document
from .services import generate_embeddings_for_dataset, ingest_json_and_create_dataset


def dataset_list(request):
    datasets = Dataset.objects.order_by("-created_at")
    return render(request, "embeddings/datasets_list.html", {"datasets": datasets})

def dataset_detail(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    # dataset.refresh_from_db()

    doc_qs = dataset.documents.all().order_by("id")

    paginator = Paginator(doc_qs, 50)
    page_number = request.GET.get("page", 1)

    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)

    context = {
        "dataset": dataset,
        "page_obj": page_obj,
    }
    return render(request, "embeddings/dataset_detail.html", context)

def upload_dataset(request):
    if request.method == "POST":
        form = UploadDatasetForm(request.POST, request.FILES)
        if form.is_valid():
            name = form.cleaned_data["name"]
            description = form.cleaned_data["description"]
            model_name = form.cleaned_data["model_name"]
            file_obj = form.cleaned_data["file"]

            try:
                dataset = ingest_json_and_create_dataset(
                    file_obj=file_obj,
                    name=name,
                    description=description,
                    model_name=model_name,
                )
            except ValueError as e:
                form.add_error("file", str(e))
            else:
                messages.success(request, f"Dataset '{dataset.name}' successfully created.")
                return redirect("embeddings:dataset_detail", pk=dataset.pk)
    else:
        form = UploadDatasetForm()

    return render(request, "embeddings/upload_dataset.html", {"form": form})

def start_generation(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)

    # Lancio in thread per evitare timeout HTTP
    t = threading.Thread(
        target=generate_embeddings_for_dataset,
        args=(dataset.pk,),
        daemon=True
    )
    t.start()

    messages.success(
        request,
        f"Embeddings generation started for '{dataset.name}' in background. Refresh page to monitor progress.",
    )
    return redirect("embeddings:dataset_detail", pk=dataset.pk)

def document_detail(request, doc_id):
    document = get_object_or_404(Document, pk=doc_id)
    dataset = document.dataset
    context = {
        "document": document,
        "dataset": dataset,
    }
    return render(request, "embeddings/document_detail.html", context)

def delete_dataset(request, pk):
    """Vista per eliminare un dataset e i suoi documenti."""
    dataset = get_object_or_404(Dataset, pk=pk)
    if request.method == "POST":
        name = dataset.name

        # Delete ChromaDB collection
        try:
            from search.client import is_available
            if is_available():
                from search.collections import delete_document_collection
                delete_document_collection(dataset.id)
        except Exception:
            pass

        dataset.delete()
        messages.success(request, f"Dataset '{name}' has been deleted.")
        return redirect("embeddings:dataset_list")

    # Fallback se chiamata via GET (redirect senza fare nulla)
    return redirect("embeddings:dataset_detail", pk=pk)

