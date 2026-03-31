# sae/views.py
import threading

from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render

from .forms import SAERunForm
from .models import SAERun
from .trainer import train_sae_run


def run_list(request):
    runs = SAERun.objects.all().order_by('-created_at')
    return render(request, 'sae/run_list.html', {'runs': runs})

def create_run(request):
    if request.method == 'POST':
        form = SAERunForm(request.POST)
        if form.is_valid():
            run = form.save(commit=False)

            # Detect input dimension from dataset
            first_doc = run.dataset.documents.filter(status='done').first()
            if not first_doc or not first_doc.embedding:
                messages.error(request, "Selected dataset has no valid embeddings.")
                return render(request, 'sae/create_run.html', {'form': form})

            run.input_dim = len(first_doc.embedding)
            run.status = 'queued'
            run.save()

            messages.success(request, f"Run #{run.id} created successfully.")
            return redirect('sae:run_detail', pk=run.pk)
    else:
        form = SAERunForm()

    return render(request, 'sae/create_run.html', {'form': form})

def run_detail(request, pk):
    run = get_object_or_404(SAERun, pk=pk)
    return render(request, 'sae/run_detail.html', {'run': run})

def start_run(request, pk):
    run = get_object_or_404(SAERun, pk=pk)

    if run.status == 'running':
        messages.warning(request, "Training is already running.")
        return redirect('sae:run_detail', pk=pk)

    # Launch Training in a Thread
    thread = threading.Thread(target=train_sae_run, args=(run.id,), daemon=True)
    thread.start()

    messages.info(request, "Training started in background...")
    return redirect('sae:run_detail', pk=pk)
