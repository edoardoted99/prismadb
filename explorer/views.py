import threading

import numpy as np
import torch
from django.contrib import messages
from django.db.models import Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt

from embeddings.embedders import get_embedder
from embeddings.models import Dataset, Document
from sae.models import SAERun

from .family_builder import build_feature_families
from .forms import InterpretForm
from .interpreter import (
    interpret_single_feature,
    load_sae_model,
    run_interpretation_pipeline,
    zscore_transform,
)
from .llm_utils import DEFAULT_SYSTEM_PROMPT, get_ollama_models
from .models import FeatureFamily, Interpretation, SAEFeature
from .statistics import calculate_statistics_pipeline


def index(request):
    runs = SAERun.objects.filter(status='completed').order_by('-created_at')
    for run in runs:
        run.interpreted_count = run.features.count()
    return render(request, 'explorer/index.html', {'runs': runs})

def start_interpretation(request):
    if request.method == 'POST':
        form = InterpretForm(request.POST)
        if form.is_valid():
            run = form.cleaned_data['run']
            n_feat = form.cleaned_data['n_features']
            model_id = form.cleaned_data['ollama_model']
            sys_prompt = form.cleaned_data['system_prompt']
            k_pos = form.cleaned_data['k_positive']
            k_neg = form.cleaned_data['k_negative']
            temp = form.cleaned_data['temperature']

            t = threading.Thread(
                target=run_interpretation_pipeline,
                args=(run.id, n_feat, model_id, sys_prompt, k_pos, k_neg, temp),
                daemon=True,
                name='run_interpretation_pipeline'
            )
            t.start()

            messages.success(request, f"Interpretation started for Run #{run.id} (Temp: {temp}). It runs in background.")
            return redirect('explorer:feature_list', run_id=run.id)
    else:
        last_run = SAERun.objects.filter(status='completed').order_by('-created_at').first()
        form = InterpretForm(initial={'run': last_run} if last_run else None)

    return render(request, 'explorer/start_interpret.html', {'form': form})

from django.core.paginator import Paginator


def feature_list(request, run_id):
    run = get_object_or_404(SAERun, pk=run_id)
    query = request.GET.get('q', '')

    features = run.features.all()

    if query:
        features = features.filter(
            Q(label__icontains=query) |
            Q(description__icontains=query) |
            Q(feature_index__icontains=query)
        )

    features = features.order_by('-max_activation')

    # Paginazione
    paginator = Paginator(features, 250)  # 50 features per pagina
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'explorer/feature_list.html', {
        'run': run,
        'page_obj': page_obj,
        'query': query
    })
def feature_detail(request, run_id, feature_index):
    ollama_models = get_ollama_models()
    run = get_object_or_404(SAERun, pk=run_id)

    # Crea feature se non esiste
    feature, _ = SAEFeature.objects.get_or_create(
        run=run,
        feature_index=feature_index,
        defaults={'label': f'Feature {feature_index}', 'description': 'Not interpreted yet.'}
    )

    # Gestione Versioni
    version_id = request.GET.get('version')
    current_interp = getattr(feature, 'active_interpretation', None)

    if version_id:
        try:
            current_interp = feature.interpretations.get(pk=version_id)
        except (Interpretation.DoesNotExist, AttributeError):
            pass

    history = feature.interpretations.all().order_by('-created_at') if hasattr(feature, 'interpretations') else []

    # --- FIX ISTOGRAMMA: Calcolo il VERO massimo ---
    max_hist_count = 1
    if feature.activation_histogram and 'counts' in feature.activation_histogram:
        counts = feature.activation_histogram['counts']
        if counts and len(counts) > 0:
            max_hist_count = max(counts)
            if max_hist_count == 0: max_hist_count = 1
    # -----------------------------------------------

    # --- FIX ETICHETTE: Sostituisco "Feature N" con la Label reale ---
    related_ids = set()

    # Raccogli ID da correlazioni e co-occorrenze
    if feature.correlated_features:
        related_ids.update(c['index'] for c in feature.correlated_features)

    if feature.co_occurring_features:
        related_ids.update(c['index'] for c in feature.co_occurring_features)

    # Recupera le etichette dal DB
    known_labels = {}
    if related_ids:
        # Prendo 'feature_index' e 'label' solo se la label non è vuota o "Feature X"
        feats_qs = SAEFeature.objects.filter(run=run, feature_index__in=related_ids).exclude(label="")
        known_labels = {f.feature_index: f.label for f in feats_qs}

    # Applica le etichette alle strutture dati in memoria (senza salvare su DB)
    if feature.correlated_features:
        for c in feature.correlated_features:
            if c['index'] in known_labels:
                c['label'] = known_labels[c['index']]

    if feature.co_occurring_features:
        for c in feature.co_occurring_features:
            if c['index'] in known_labels:
                c['label'] = known_labels[c['index']]
    # -----------------------------------------------------------------

    # --- SCATTER PLOT DATA: Mean vs Variance per tutte le feature ---
    # Recuperiamo solo i campi necessari per alleggerire la query
    all_features_stats = SAEFeature.objects.filter(run=run).values(
        'feature_index', 'mean_activation', 'variance_activation', 'label'
    )

    scatter_data = []
    for f_stat in all_features_stats:
        # Includiamo solo se i dati esistono (potrebbero essere None se stats non calcolate)
        if f_stat['mean_activation'] is not None and f_stat['variance_activation'] is not None:
            scatter_data.append({
                'id': f_stat['feature_index'],
                'x': f_stat['mean_activation'],
                'y': f_stat['variance_activation'],
                'label': f_stat['label'] if f_stat['label'] else f"Feature {f_stat['feature_index']}"
            })
    return render(request, 'explorer/feature_detail.html', {
            'feature': feature,
            'run': feature.run,
            'current_interp': current_interp,
            'history': history,
            'max_hist_count': max_hist_count,

            # 2. Passali al context (Mancava anche questo!)
            'ollama_models': ollama_models,
            'ollama_status': 'online' if ollama_models else 'offline',
            'default_system_prompt': DEFAULT_SYSTEM_PROMPT,
            'scatter_data': scatter_data, # Dati per il grafico
        })

from django.utils.safestring import mark_safe


def reinterpret_feature(request, run_id, feature_index):
    feature = get_object_or_404(SAEFeature, run_id=run_id, feature_index=feature_index)

    if request.method == "POST":
        model_id = request.POST.get("ollama_model", "qwen2.5:14b")
        prompt = request.POST.get("system_prompt", "")
        try:
            temp = float(request.POST.get("temperature", 0.2))
            k_pos = int(request.POST.get("k_positive", 5))
            k_neg = int(request.POST.get("k_negative", 5))
        except ValueError:
            temp = 0.2; k_pos = 5; k_neg = 5

        # Run in background thread
        def run_interpretation_task():
            interpret_single_feature(feature.id, model_id, prompt, k_pos, k_neg, temp)

        t = threading.Thread(target=run_interpretation_task)
        t.start()

        status_url = reverse('explorer:system_status')
        msg = mark_safe(f'Interpretation started in background. <a href="{status_url}" target="_blank" class="btn btn-sm btn-outline-light ms-2">Check Status</a>')
        messages.success(request, msg)

    return redirect('explorer:feature_detail', run_id=run_id, feature_index=feature_index)



def calculate_stats(request, run_id):
    # Check if interpretation is already running to avoid OOM/Crash
    for t in threading.enumerate():
        if t.name == 'run_interpretation_pipeline' and t.is_alive():
            messages.error(request, "Cannot start statistics calculation while interpretation is running. Please wait for it to finish.")
            return redirect(request.META.get('HTTP_REFERER', 'explorer:feature_list'))

    # Lancia il calcolo in background
    t = threading.Thread(target=calculate_statistics_pipeline, args=(run_id,), daemon=True, name='calculate_statistics_pipeline')
    t.start()

    status_url = reverse('explorer:system_status')
    msg = mark_safe(f'Statistics calculation started in background. <a href="{status_url}" target="_blank" class="btn btn-sm btn-outline-light ms-2">Check Status</a>')
    messages.info(request, msg)

    # LOGICA DI REDIRECT MIGLIORATA
    # Se la richiesta ha un 'next', usalo
    next_url = request.GET.get('next')

    # Se no, prova a tornare alla pagina da cui sei venuto (Referer)
    if not next_url:
        next_url = request.META.get('HTTP_REFERER')

    # Se ancora nulla, torna alla lista feature di default
    if not next_url:
        next_url = reverse('explorer:feature_list', args=[run_id])

    return redirect(next_url)

def build_families(request, run_id):
    """Trigger per costruire le famiglie"""
    # Lancia in thread per non bloccare
    t = threading.Thread(target=build_feature_families, args=(run_id,), daemon=True)
    t.start()
    messages.info(request, "Feature Family construction started. Refresh in a moment.")
    return redirect('explorer:family_list', run_id=run_id)

def family_list(request, run_id):
    run = get_object_or_404(SAERun, pk=run_id)
    families = FeatureFamily.objects.filter(run=run)

    # Search Functionality
    query = request.GET.get('q')
    if query:
        families = families.filter(
            Q(parent_feature__label__icontains=query) |
            Q(children_features__label__icontains=query)
        ).distinct()

    # Inverted Index for "By Child" view
    child_map = {}
    for fam in families:
        for child in fam.children_features.all():
            if child not in child_map:
                child_map[child] = []
            child_map[child].append(fam)

    inverted_families = []
    for child, fams in child_map.items():
        inverted_families.append({'child': child, 'families': fams})

    # Sort by child feature index
    inverted_families.sort(key=lambda x: x['child'].feature_index)

    return render(request, 'explorer/family_list.html', {
        'run': run,
        'families': families,
        'inverted_families': inverted_families
    })

def document_analyzer(request):
    available_models = get_ollama_models()
    dataset_id = request.GET.get('dataset_id')
    run_id = request.GET.get('run_id')
    doc_id = request.GET.get('doc_id')
    doc_query = request.GET.get('doc_q', '').strip()

    context = {
        'datasets': Dataset.objects.all(),
        'selected_dataset': None,
        'selected_run': None,
        'selected_doc': None,
        'activations': [],
        'run_list': [],
        'documents_list': [],
        'similar_docs': [],
        'embedding_norm': 0.0,
        'ollama_models': available_models,
        'ollama_status': 'online' if available_models else 'offline'
    }

    # 1. Dataset Selezionato
    if dataset_id:
        ds = get_object_or_404(Dataset, pk=dataset_id)
        context['selected_dataset'] = ds
        context['run_list'] = ds.sae_runs.filter(status='completed')

    # 2. Run Selezionata (STEP FONDAMENTALE)
    if run_id and dataset_id:
        run = get_object_or_404(SAERun, pk=run_id)
        context['selected_run'] = run

        # MOSTRA LISTA DOCUMENTI SOLO ORA (Se non c'è un doc specifico)
        if not doc_id:
            docs_qs = Document.objects.filter(dataset_id=dataset_id)
            if doc_query:
                docs_qs = docs_qs.filter(Q(text__icontains=doc_query) | Q(external_id__icontains=doc_query))
            # Carichiamo un po' più documenti per facilitare la ricerca
            context['documents_list'] = docs_qs.order_by('id')[:200]

    # 3. Analisi Documento (Solo se Dataset + Run + Doc sono presenti)
    if dataset_id and run_id and doc_id:
        doc = get_object_or_404(Document, pk=doc_id)

        # Check coerenza
        if str(doc.dataset_id) == str(dataset_id):
            context['selected_doc'] = doc

            # --- A. Embedding Info & Similarity ---
            from search.bulk_ops import get_document_embedding
            doc_embedding = get_document_embedding(int(dataset_id), doc.id)

            if doc_embedding:
                try:
                    target_emb = np.array(doc_embedding)
                    context['embedding_norm'] = float(np.linalg.norm(target_emb))

                    # kNN similarity search via ChromaDB
                    from search.queries import search_similar_documents
                    knn_results = search_similar_documents(
                        int(dataset_id), doc_embedding, k=5, exclude_id=doc.id
                    )

                    if knn_results:
                        for hit in knn_results:
                            try:
                                sim_doc = Document.objects.get(pk=hit['django_id'])
                                context['similar_docs'].append({
                                    'doc': sim_doc,
                                    'score': hit['score'],
                                })
                            except Document.DoesNotExist:
                                pass

                except Exception as e:
                    print(f"[Sim Error] {e}")

            # --- B. SAE Inference ---
            if doc_embedding and context['selected_run']:
                try:
                    device = "cpu"
                    model, mean, std = load_sae_model(run, device)

                    if model:
                        emb_tensor = torch.tensor([doc_embedding], dtype=torch.float32).to(device)
                        if mean is not None:
                            emb_tensor = zscore_transform(emb_tensor, mean, std)

                        with torch.no_grad():
                            _, _, h_topk = model(emb_tensor)
                            acts = h_topk[0]

                        non_zero = torch.nonzero(acts > 0.0001).flatten()
                        values = acts[non_zero].tolist()
                        indices = non_zero.tolist()

                        # Recupero info features
                        features_db = SAEFeature.objects.filter(
                            run=run, feature_index__in=indices
                        ).select_related('active_interpretation')
                        features_map = {f.feature_index: f for f in features_db}

                        analyzed_features = []
                        for idx, val in zip(indices, values):
                            feat_obj = features_map.get(idx)
                            label = f"Feature #{idx}"
                            desc = "" # Descrizione vuota di default per pulizia visiva

                            if feat_obj:
                                if hasattr(feat_obj, 'active_interpretation') and feat_obj.active_interpretation:
                                    label = feat_obj.active_interpretation.label
                                    desc = feat_obj.active_interpretation.description
                                elif feat_obj.label:
                                    label = feat_obj.label
                                    desc = feat_obj.description

                            analyzed_features.append({
                                'index': idx,
                                'activation': val,
                                'label': label,
                                'description': desc,
                                'db_id': feat_obj.id if feat_obj else None
                            })

                        # Sort per attivazione
                        analyzed_features.sort(key=lambda x: x['activation'], reverse=True)
                        context['activations'] = analyzed_features

                except Exception as e:
                    print(f"[SAE Error] {e}")

    return render(request, 'explorer/document_analyzer.html', context)

# --- SYSTEM STATUS & PROCESS MANAGEMENT ---
import ctypes
import time

from django.conf import settings

from .task_status import TASK_PROGRESS


def system_status(request):
    from project.utils import get_setting
    return render(request, 'explorer/system_status.html', {
        'ollama_url': get_setting('ollama_base_url'),
        'prismadb_version': settings.PRISMADB_VERSION,
    })

def get_logs(request):
    """Returns the last N lines of the log file."""
    log_file = settings.BASE_DIR / 'debug.log'
    last_line = int(request.GET.get('last_line', 0))
    lines = []
    current_line_count = 0

    if log_file.exists():
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            current_line_count = len(all_lines)
            # If client is requesting from 0, give last 1000.
            # If client has a specific index, give everything after that.
            if last_line == 0:
                lines = all_lines[-1000:]
            elif last_line < current_line_count:
                lines = all_lines[last_line:]

    return JsonResponse({
        'lines': lines,
        'last_line': current_line_count
    })

def get_threads(request):
    """Returns a list of active threads."""
    threads = []
    for t in threading.enumerate():
        # Skip MainThread to avoid showing the server itself as a task
        if t.name == 'MainThread':
            continue

        status = 'Running' if t.is_alive() else 'Stopped'

        # Check for progress info
        progress_info = TASK_PROGRESS.get(t.ident, {})
        progress_pct = progress_info.get('progress', 0)
        progress_msg = progress_info.get('message', '')
        start_time = progress_info.get('start_time', None)

        duration_str = '-'
        if start_time:
            duration = time.time() - start_time
            # Format duration as HH:MM:SS
            m, s = divmod(duration, 60)
            h, m = divmod(m, 60)
            duration_str = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

        threads.append({
            'name': t.name,
            'ident': t.ident,
            'daemon': t.daemon,
            'status': status,
            'progress': progress_pct,
            'message': progress_msg,
            'duration': duration_str,
            'start_time': start_time
        })

    return JsonResponse({'threads': threads})

def get_system_stats(request):
    """Returns system resource usage (CPU, RAM, GPU)."""
    import psutil
    import torch

    # CPU & RAM
    cpu_percent = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    ram_percent = ram.percent
    ram_used_gb = ram.used / (1024**3)
    ram_total_gb = ram.total / (1024**3)

    # GPU
    gpu_stats = {'available': False, 'name': 'N/A', 'memory_used': 0, 'memory_total': 0, 'utilization': 0, 'temperature': 0}

    if torch.cuda.is_available():
        gpu_stats['available'] = True
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            gpu_stats['name'] = name
            gpu_stats['memory_used'] = info.used / (1024**3)
            gpu_stats['memory_total'] = info.total / (1024**3)

            # Utilization & Temp
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_stats['utilization'] = util.gpu
            except Exception:
                gpu_stats['utilization'] = 0

            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_stats['temperature'] = temp
            except Exception:
                gpu_stats['temperature'] = 0

            pynvml.nvmlShutdown()
        except ImportError:
            # Fallback to torch if pynvml not installed
            gpu_stats['name'] = torch.cuda.get_device_name(0)
            gpu_stats['memory_used'] = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_stats['memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_stats['utilization'] = 0
            gpu_stats['temperature'] = 0
        except Exception as e:
            print(f"Error getting NVML stats: {e}")
            gpu_stats['name'] = torch.cuda.get_device_name(0)
            gpu_stats['memory_used'] = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_stats['memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_stats['utilization'] = 0
            gpu_stats['temperature'] = 0

    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        gpu_stats['available'] = True
        gpu_stats['name'] = 'Apple MPS'

        try:
            # prova prima la memoria che vede il driver (più simile a nvtop)
            used_bytes = torch.mps.driver_allocated_memory()
            if used_bytes == 0:
                # fallback: solo tensori, se per qualche motivo driver_allocated_memory è 0
                used_bytes = torch.mps.current_allocated_memory()

            gpu_stats['memory_used'] = used_bytes / (1024**3)

            # su Apple la memoria è unificata → totale "GPU" = RAM totale
            gpu_stats['memory_total'] = ram_total_gb

        except Exception as e:
            print(f"MPS Stats Error: {e}")
            gpu_stats['memory_used'] = 0
            gpu_stats['memory_total'] = ram_total_gb

        # Non c'è una API pubblica per % di utilizzo MPS: la lasci a 0 oppure
        # la togli dall'UI se sei su Apple.
        gpu_stats['utilization'] = 0
        gpu_stats['temperature'] = 0

    return JsonResponse({
        'cpu': cpu_percent,
        'ram': {
            'percent': ram_percent,
            'used_gb': f"{ram_used_gb:.1f}",
            'total_gb': f"{ram_total_gb:.1f}"
        },
        'gpu': gpu_stats
    })


def get_services_status(request):
    """Check connectivity to Ollama and ChromaDB."""
    import requests as req

    from project.utils import get_setting

    # Ollama
    ollama_url = get_setting('ollama_base_url')
    ollama_ok = False
    ollama_models = []
    try:
        resp = req.get(f"{ollama_url}/api/tags", timeout=5)
        if resp.status_code == 200:
            ollama_ok = True
            data = resp.json()
            ollama_models = [m['name'] for m in data.get('models', [])]
    except Exception:
        pass

    # ChromaDB
    chromadb_ok = False
    chromadb_info = {}
    try:
        from search.client import get_client, is_available
        chromadb_ok = is_available()
        if chromadb_ok:
            client = get_client()
            chromadb_info = {
                'collections': len(client.list_collections()),
                'path': str(settings.PRISMADB_HOME / "chromadb_data"),
            }
    except Exception:
        pass

    return JsonResponse({
        'ollama': {
            'connected': ollama_ok,
            'url': ollama_url,
            'models': ollama_models,
        },
        'chromadb': {
            'connected': chromadb_ok,
            **chromadb_info,
        },
        'version': settings.PRISMADB_VERSION,
    })


@csrf_exempt
def update_ollama_url(request):
    """Update the Ollama base URL at runtime."""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)

    import json
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON'}, status=400)

    new_url = body.get('url', '').strip().rstrip('/')
    if not new_url:
        return JsonResponse({'success': False, 'error': 'URL is required'}, status=400)

    from project.utils import set_setting
    set_setting('ollama_base_url', new_url)

    # Test connection
    import requests as req
    connected = False
    try:
        resp = req.get(f"{new_url}/api/tags", timeout=5)
        connected = resp.status_code == 200
    except Exception:
        pass

    return JsonResponse({
        'success': True,
        'url': new_url,
        'connected': connected,
    })


def chromadb_status(request):
    """Check ChromaDB status."""
    connected = False
    info = {}
    try:
        from search.client import get_client, is_available
        connected = is_available()
        if connected:
            client = get_client()
            info['collections'] = len(client.list_collections())
            info['path'] = str(settings.PRISMADB_HOME / "chromadb_data")
    except Exception:
        pass

    return JsonResponse({
        'connected': connected,
        **info,
    })


@csrf_exempt
def kill_thread(request, ident):
    """Attempts to kill a thread by raising an exception in it."""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)

    if not request.user.is_staff: # Basic protection
         return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)

    target_tid = int(ident)

    # Check if thread exists
    found = False
    for t in threading.enumerate():
        if t.ident == target_tid:
            found = True
            break

    if not found:
        return JsonResponse({'success': False, 'error': 'Thread not found'})

    # Ctypes magic to raise exception in thread
    try:
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(target_tid),
            ctypes.py_object(SystemExit)
        )
        if res == 0:
            return JsonResponse({'success': False, 'error': 'Invalid thread ID'})
        elif res > 1:
            # If it returns a number greater than 1, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect
            ctypes.pythonapi.PyThreadState_SetAsyncExc(target_tid, 0)
            return JsonResponse({'success': False, 'error': 'Failed to kill thread (state rollback)'})

        return JsonResponse({'success': True, 'message': f'Signal sent to thread {target_tid}'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@csrf_exempt
def stop_interpretation(request, run_id):
    """Signals the interpretation pipeline to stop gracefully."""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)

    from .interpreter import TASK_CONTROL
    TASK_CONTROL[run_id] = 'STOP'

    return JsonResponse({'success': True, 'message': 'Pause signal sent. The process will stop after the current feature.'})

import csv
import os

from django.http import FileResponse, HttpResponse


def export_feature_statistics(request, run_id):
    """
    Exports statistics for all features in a run as a CSV file.
    """
    run = get_object_or_404(SAERun, pk=run_id)
    features = run.features.all().order_by('feature_index')

    response = HttpResponse(
        content_type='text/csv',
        headers={'Content-Disposition': f'attachment; filename="run_{run_id}_stats.csv"'},
    )

    writer = csv.writer(response)
    # Header
    writer.writerow([
        'Feature Index',
        'Label',
        'Description',
        'Density',
        'Max Activation',
        'Mean Activation',
        'Variance Activation'
    ])

    # Data
    for feature in features:
        # Use active interpretation label/desc if available, otherwise fallback
        label = feature.label
        description = feature.description

        if hasattr(feature, 'active_interpretation') and feature.active_interpretation:
            label = feature.active_interpretation.label
            description = feature.active_interpretation.description

        writer.writerow([
            feature.feature_index,
            label,
            description,
            feature.density,
            feature.max_activation,
            feature.mean_activation,
            feature.variance_activation
        ])

    return response

def download_db(request):
    """
    Invisible route to download the sqlite database for debugging.
    """
    db_path = settings.DATABASES['default']['NAME']
    if os.path.exists(db_path):
        return FileResponse(open(db_path, 'rb'), as_attachment=True, filename='db.sqlite3')
    else:
        return HttpResponse("Database file not found", status=404)

def download_sae_weights(request, run_id):
    """
    Download the SAE weights (.pt) file for a specific run.
    """
    run = get_object_or_404(SAERun, pk=run_id)

    if not run.weights_file:
        messages.error(request, "Weights file not found for this run.")
        return redirect('explorer:family_list', run_id=run_id)

    try:
        return FileResponse(run.weights_file.open('rb'), as_attachment=True, filename=f"sae_weights_run_{run.id}.pt")
    except FileNotFoundError:
        messages.error(request, "Weights file missing from storage.")
        return redirect('explorer:family_list', run_id=run_id)

def export_document_activations(request, run_id):
    """
    Exports sparse activations for all documents in a dataset using the SAE run.
    Yields CSV rows as a stream to avoid memory issues.
    """
    import json

    from django.http import StreamingHttpResponse

    run = get_object_or_404(SAERun, pk=run_id)
    dataset = run.dataset

    # Define a generator for streaming
    def csv_generator():
        yield "doc_id,text_snippet,sparse_vector_json\n"

        # Load Model once
        device = "cpu"
        model, mean, std = load_sae_model(run, device)
        if not model:
            yield "ERROR: SAE Model not found.\n"
            return

        from search.bulk_ops import scroll_documents_in_batches

        for batch_data in scroll_documents_in_batches(dataset.id, batch_size=100,
                                                       fields=['django_id', 'embedding']):
            for doc_data in batch_data:
                embedding = doc_data.get('embedding')
                if embedding is None:
                    continue

                django_id = doc_data['django_id']
                try:
                    doc = Document.objects.get(pk=django_id)
                except Document.DoesNotExist:
                    continue

                try:
                    emb_tensor = torch.tensor([embedding], dtype=torch.float32)
                    if mean is not None:
                        emb_tensor = zscore_transform(emb_tensor, mean, std)

                    with torch.no_grad():
                        _, _, h_topk = model(emb_tensor)
                        acts = h_topk[0]

                    non_zero = torch.nonzero(acts > 0.0001).flatten()
                    values = acts[non_zero].tolist()
                    indices = non_zero.tolist()

                    sparse_vector = {idx: round(val, 4) for idx, val in zip(indices, values)}
                    sparse_json = json.dumps(sparse_vector)

                    truncate_len = getattr(settings, 'EXPLORER_DOC_TRUNCATION_LIMIT', 500)
                    text_snippet = doc.text[:truncate_len].replace('"', '""').replace('\n', ' ')
                    escaped_json = sparse_json.replace('"', '""')

                    yield f'{doc.external_id},"{text_snippet}","{escaped_json}"\n'

                except Exception as e:
                    print(f"Error exporting doc {doc.id}: {e}")
                    continue

    response = StreamingHttpResponse(
        csv_generator(),
        content_type='text/csv',
        headers={'Content-Disposition': f'attachment; filename="run_{run.id}_activations.csv"'},
    )
    return response

def inference_view(request):
    runs = SAERun.objects.filter(status='completed').order_by('-created_at')

    context = {
        'runs': runs,
        'selected_run': None,
        'input_text': '',
        'analyzed_features': [],
        'embedding_norm': 0.0,
    }

    if request.method == 'POST':
        run_id = request.POST.get('run_id')
        text = request.POST.get('text', '').strip()

        if run_id and text:
            run = get_object_or_404(SAERun, pk=run_id)
            context['selected_run'] = run
            context['input_text'] = text

            try:
                # 1. Generate Embedding
                model_name = run.dataset.model_name
                embedder_cls = get_embedder(model_name)
                # embed_texts returns a list of lists
                embeddings = embedder_cls.embed_texts([text])

                if embeddings and len(embeddings) > 0 and len(embeddings[0]) > 0:
                    embedding = embeddings[0]
                    emb_tensor = torch.tensor([embedding], dtype=torch.float32)
                    context['embedding_norm'] = float(torch.norm(emb_tensor))

                    # 2. Load SAE & Inference
                    device = "cpu" # Force CPU for inference view to be safe/simple
                    model, mean, std = load_sae_model(run, device)

                    if model:
                        if mean is not None:
                            emb_tensor = zscore_transform(emb_tensor, mean, std)

                        with torch.no_grad():
                            _, _, h_topk = model(emb_tensor)
                            acts = h_topk[0]

                        # 3. Filter & Format Results
                        # Filter > 0.0001 to remove noise
                        non_zero = torch.nonzero(acts > 0.0001).flatten()
                        values = acts[non_zero].tolist()
                        indices = non_zero.tolist()

                        # Get Feature Details from DB
                        features_db = SAEFeature.objects.filter(
                            run=run, feature_index__in=indices
                        ).select_related('active_interpretation')
                        features_map = {f.feature_index: f for f in features_db}

                        analyzed_features = []
                        max_val = max(values) if values else 1.0

                        for idx, val in zip(indices, values):
                            feat_obj = features_map.get(idx)
                            label = ""
                            desc = ""

                            if feat_obj:
                                if hasattr(feat_obj, 'active_interpretation') and feat_obj.active_interpretation:
                                    label = feat_obj.active_interpretation.label
                                    desc = feat_obj.active_interpretation.description
                                elif feat_obj.label:
                                    label = feat_obj.label
                                    desc = feat_obj.description

                            analyzed_features.append({
                                'index': idx,
                                'activation': val,
                                'activation_pct': (val / max_val) * 100,
                                'label': label,
                                'description': desc
                            })

                        # Histogram Data
                        if values:
                            counts, bin_edges = np.histogram(values, bins=20)
                            # Use bin centers for plotting
                            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                            context['hist_x'] = bin_centers.tolist()
                            context['hist_y'] = counts.tolist()

                        # Sort by activation desc
                        analyzed_features.sort(key=lambda x: x['activation'], reverse=True)
                        context['analyzed_features'] = analyzed_features

            except Exception as e:
                messages.error(request, f"Inference Error: {str(e)}")
                print(f"[Inference Error] {e}")

    return render(request, 'explorer/inference.html', context)
