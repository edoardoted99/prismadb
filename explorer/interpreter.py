import torch
import heapq
import time
import random
import traceback
import logging
from django.conf import settings
from sae.models import SAERun
from .models import SAEFeature, Interpretation
from sae.modules import SAE, SAEConfig, zscore_transform
from .llm_utils import get_ollama_response
from .task_status import TASK_PROGRESS
from project.utils import get_device
import threading

logger = logging.getLogger(__name__)

def load_sae_model(run, device):
    """Helper per caricare il modello SAE"""
    logger.info(f"[Interpreter] Loading SAE model from: {run.weights_file.path}...")
    if not run.weights_file:
        logger.error("[Interpreter] ERROR: No weights file associated with this Run.")
        return None, None, None
    
    try:
        checkpoint = torch.load(run.weights_file.path, map_location=device)
        cfg_dict = checkpoint['config']
        cfg_dict['device'] = device
        
        cfg = SAEConfig(**cfg_dict)
        model = SAE(cfg).to(device)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        
        mean = checkpoint['zscore_mean'].to(device)
        std = checkpoint['zscore_std'].to(device)
        
        logger.info(f"[Interpreter] Model loaded successfully. Latent dim: {cfg.d_latent}")
        return model, mean, std
    except Exception as e:
        logger.error(f"[Interpreter] Failed to load model: {e}", exc_info=True)
        return None, None, None

def scan_single_feature_examples(run, feature_index, k=10):
    """
    Scansiona il dataset SOLO per una specifica feature.
    Uses OpenSearch scroll if available, otherwise falls back to SQLite.
    """
    device = get_device()
    logger.info(f"[Interpreter] Single-scan for feature {feature_index} on {device}...")

    model, mean, std = load_sae_model(run, device)
    if not model:
        return []

    top_heap = []
    batch_size = 512
    truncate_len = getattr(settings, 'EXPLORER_DOC_TRUNCATION_LIMIT', 500)
    processed = 0

    # Try OpenSearch scroll first
    use_opensearch = False
    try:
        from search.client import is_available
        if is_available():
            from search.bulk_ops import scroll_documents_in_batches, count_documents
            total = count_documents(run.dataset_id)
            use_opensearch = True
            logger.info(f"[Interpreter] Scanning {total} documents via OpenSearch for feature {feature_index}...")
    except Exception:
        use_opensearch = False

    if use_opensearch:
        for batch_data in scroll_documents_in_batches(run.dataset_id, batch_size=batch_size,
                                                       fields=['django_id', 'text', 'embedding']):
            embeddings = [d['embedding'] for d in batch_data if d.get('embedding')]
            if not embeddings:
                processed += len(batch_data)
                continue

            try:
                X_batch = torch.tensor(embeddings, dtype=torch.float32).to(device)
                X_batch = zscore_transform(X_batch, mean, std)

                with torch.no_grad():
                    acts = model.encode(X_batch)
                    feat_acts = acts[:, feature_index].cpu()

                indices = torch.nonzero(feat_acts > 0.001).flatten()

                for idx in indices:
                    val = feat_acts[idx].item()
                    doc = batch_data[idx.item()]
                    text = doc.get('text', '')[:truncate_len]

                    if len(top_heap) < k:
                        heapq.heappush(top_heap, (val, doc['django_id'], text))
                    elif val > top_heap[0][0]:
                        heapq.heapreplace(top_heap, (val, doc['django_id'], text))
            except Exception as e:
                logger.error(f"[Interpreter] Error in batch processing: {e}")

            processed += len(batch_data)
            if processed % 1000 == 0:
                logger.info(f"[Interpreter] Scanned {processed}/{total} docs...")
    else:
        # Fallback: SQLite
        doc_qs = run.dataset.documents.filter(status='done').order_by('id')
        total = doc_qs.count()
        logger.info(f"[Interpreter] Scanning {total} documents via SQLite for feature {feature_index}...")

        for offset in range(0, total, batch_size):
            batch_docs = list(doc_qs[offset:offset+batch_size])
            if not batch_docs: break

            embeddings = [d.embedding for d in batch_docs if d.embedding]
            if not embeddings:
                processed += len(batch_docs)
                continue

            try:
                X_batch = torch.tensor(embeddings, dtype=torch.float32).to(device)
                X_batch = zscore_transform(X_batch, mean, std)

                with torch.no_grad():
                    acts = model.encode(X_batch)
                    feat_acts = acts[:, feature_index].cpu()

                indices = torch.nonzero(feat_acts > 0.001).flatten()

                for idx in indices:
                    val = feat_acts[idx].item()
                    doc = batch_docs[idx.item()]

                    if len(top_heap) < k:
                        heapq.heappush(top_heap, (val, doc.id, doc.text[:truncate_len]))
                    elif val > top_heap[0][0]:
                        heapq.heapreplace(top_heap, (val, doc.id, doc.text[:truncate_len]))
            except Exception as e:
                logger.error(f"[Interpreter] Error in batch processing: {e}")

            processed += len(batch_docs)
            if processed % 1000 == 0:
                logger.info(f"[Interpreter] Scanned {processed}/{total} docs...")

    results = sorted(top_heap, key=lambda x: x[0], reverse=True)
    logger.info(f"[Interpreter] Scan finished. Found {len(results)} examples > 0.001")

    formatted_docs = [
        {'id': did, 'act': float(v), 'text': txt}
        for v, did, txt in results
    ]
    return formatted_docs

def get_negative_examples(run, feature_index, k=5, model=None, mean=None, std=None):
    """
    Recupera k esempi negativi (attivazione zero o molto bassa) per una feature.
    Uses OpenSearch random_score if available, otherwise falls back to SQLite.
    """
    device = get_device()
    if not model:
        model, mean, std = load_sae_model(run, device)
        if not model: return []

    negatives = []
    attempts = 0
    max_attempts = 50
    truncate_len = getattr(settings, 'EXPLORER_DOC_TRUNCATION_LIMIT', 500)

    while len(negatives) < k and attempts < max_attempts:
        attempts += 1

        # Try OpenSearch random docs first
        random_docs_data = None
        try:
            from search.client import is_available
            if is_available():
                from search.queries import get_random_documents
                random_docs_data = get_random_documents(run.dataset_id, k=k*2)
        except Exception:
            random_docs_data = None

        if random_docs_data:
            embeddings = [d['embedding'] for d in random_docs_data if d.get('embedding')]
            if not embeddings: continue

            try:
                X_batch = torch.tensor(embeddings, dtype=torch.float32).to(device)
                X_batch = zscore_transform(X_batch, mean, std)

                with torch.no_grad():
                    acts = model.encode(X_batch)
                    feat_acts = acts[:, feature_index].cpu()

                for i, val in enumerate(feat_acts):
                    if val.item() < 0.001:
                        doc = random_docs_data[i]
                        doc_id = doc['django_id']
                        if not any(d['id'] == doc_id for d in negatives):
                            text = doc.get('text', '')[:truncate_len]
                            negatives.append({'id': doc_id, 'act': val.item(), 'text': text})
                            if len(negatives) >= k: break
            except Exception as e:
                logger.error(f"[Interpreter] Error finding negatives: {e}")
                break
        else:
            # Fallback: SQLite
            random_docs = list(run.dataset.documents.filter(status='done').order_by('?')[:k*2])
            embeddings = [d.embedding for d in random_docs if d.embedding]
            if not embeddings: continue

            try:
                X_batch = torch.tensor(embeddings, dtype=torch.float32).to(device)
                X_batch = zscore_transform(X_batch, mean, std)

                with torch.no_grad():
                    acts = model.encode(X_batch)
                    feat_acts = acts[:, feature_index].cpu()

                for i, val in enumerate(feat_acts):
                    if val.item() < 0.001:
                        doc = random_docs[i]
                        if not any(d['id'] == doc.id for d in negatives):
                            negatives.append({'id': doc.id, 'act': val.item(), 'text': doc.text[:truncate_len]})
                            if len(negatives) >= k: break
            except Exception as e:
                logger.error(f"[Interpreter] Error finding negatives: {e}")
                break

    return negatives[:k]

def interpret_single_feature(feature_id, model_name, prompt, k_pos=5, k_neg=5, temperature=0.2):
    """
    Esegue l'interpretazione per UNA singola feature.
    """
    tid = threading.get_ident()
    TASK_PROGRESS[tid] = {
        'progress': 0, 
        'message': f'Initializing interpretation for Feature {feature_id}...',
        'start_time': time.time()
    }

    logger.info(f"[Interpreter] Starting single interpretation for Feature ID {feature_id}")
    
    if not prompt:
        from .llm_utils import DEFAULT_SYSTEM_PROMPT
        prompt = DEFAULT_SYSTEM_PROMPT
    try:
        feature = SAEFeature.objects.get(pk=feature_id)
        run = feature.run
        
        # 0. REPLACE LOGIC: Delete old active interpretation if exists
        if hasattr(feature, 'active_interpretation') and feature.active_interpretation:
            logger.info(f"[Interpreter] Deleting old active interpretation #{feature.active_interpretation.id}")
            feature.active_interpretation.delete()
            feature.refresh_from_db()

        # 1. CONTROLLO ESEMPI POSITIVI
        TASK_PROGRESS[tid].update({'progress': 10, 'message': 'Scanning for positive examples...'})
        if not feature.example_docs:
            logger.info(f"[Interpreter] No cached examples. Launching scan...")
            # Fetch more examples (e.g. 50) to show in the UI, even if we only use k_pos for interpretation
            found_docs = scan_single_feature_examples(run, feature.feature_index, k=50)
            
            if not found_docs:
                logger.warning(f"[Interpreter] DEAD NEURON: Feature {feature.feature_index} has 0 activations.")
                TASK_PROGRESS[tid].update({'progress': 100, 'message': 'Failed: Dead Neuron (0 activations).'})
                return False
            
            feature.example_docs = found_docs
            feature.max_activation = found_docs[0]['act']
            feature.save()
        else:
            logger.info(f"[Interpreter] Using {len(feature.example_docs)} cached examples.")

        # 2. RECUPERO ESEMPI NEGATIVI
        TASK_PROGRESS[tid].update({'progress': 30, 'message': 'Fetching negative examples...'})
        logger.info(f"[Interpreter] Fetching {k_neg} negative examples...")
        neg_examples = get_negative_examples(run, feature.feature_index, k=k_neg)

        # 3. PROMPT
        pos_examples = feature.example_docs[:k_pos]
        prompt_text = "Positive Examples (High Activation):\n"
        for doc in pos_examples:
            clean_txt = doc['text'].replace('\n', ' ').strip()
            prompt_text += f"- [Act: {doc['act']:.2f}] {clean_txt}\n"
        
        prompt_text += "\nNegative Examples (Zero Activation):\n"
        if neg_examples:
            for doc in neg_examples:
                clean_txt = doc['text'].replace('\n', ' ').strip()
                prompt_text += f"- [Act: {doc['act']:.2f}] {clean_txt}\n"
        else:
            prompt_text += "- (No negative examples found, assuming random unrelated texts)\n"

        # 3. OLLAMA
        TASK_PROGRESS[tid].update({'progress': 50, 'message': f'Querying Ollama ({model_name})...'})
        logger.info(f"[Interpreter] Sending request to Ollama ({model_name})...")
        logger.info(f"[Interpreter] Prompt sent to Ollama:\n{prompt_text}")
        start_time = time.time()
        result = get_ollama_response(
            user_message=prompt_text, 
            system_message=prompt,
            model=model_name,
            temperature=temperature
        )
        duration = time.time() - start_time
        logger.info(f"[Interpreter] Ollama responded in {duration:.2f}s. Result: {result}")
        
        if result:
            TASK_PROGRESS[tid].update({'progress': 90, 'message': 'Saving interpretation...'})
            interp = Interpretation.objects.create(
                feature=feature,
                label=result.get('label', 'Unknown'),
                description=result.get('description', ''),
                llm_model=model_name,
                system_prompt=prompt,     # <--- CORRETTO
                temperature=temperature,
                evidence_docs={'positive': pos_examples}
            )
            
            feature.label = result.get('label', 'Unknown')
            feature.description = result.get('description', '')
            feature.active_interpretation = interp # Link new interpretation
            feature.save()
            logger.info("[Interpreter] Interpretation saved successfully.")
            
            TASK_PROGRESS[tid].update({'progress': 100, 'message': 'Done.'})
            return True
        else:
            logger.error("[Interpreter] ERROR: Ollama returned None or invalid JSON.")
            TASK_PROGRESS[tid].update({'progress': 100, 'message': 'Failed: No response from Ollama.'})
            
    except Exception as e:
        logger.error(f"[Interpreter] CRITICAL ERROR in interpret_single_feature: {e}", exc_info=True)
        TASK_PROGRESS[tid].update({'progress': 100, 'message': f'Error: {e}'})
        
    return False

# Global control dictionary: {run_id: 'STOP'}
TASK_CONTROL = {}

def run_interpretation_pipeline(run_id, features_to_analyze=50, ollama_model="qwen2.5:14b", custom_system_prompt=None, k_pos=5, k_neg=5, temp=0.2):
    """
    Pipeline Batch.
    """
    # Reset control signal for this run
    if run_id in TASK_CONTROL:
        del TASK_CONTROL[run_id]

    device = get_device()
    logger.info(f"\n{'='*50}")
    logger.info(f"[Interpreter] STARTING BATCH PIPELINE | Run #{run_id} | Device: {device}")
    logger.info(f"{'='*50}")

    # --- PROGRESS INIT ---
    tid = threading.get_ident()
    TASK_PROGRESS[tid] = {
        'progress': 0, 
        'message': 'Initializing...',
        'start_time': time.time(),
        'run_id': run_id  # Add run_id for UI control
    }
    # ---------------------

    if not custom_system_prompt:
        from .llm_utils import DEFAULT_SYSTEM_PROMPT
        custom_system_prompt = DEFAULT_SYSTEM_PROMPT
    
    try:
        run = SAERun.objects.get(pk=run_id)
        model, mean, std = load_sae_model(run, device)
        
        if not model:
            logger.error("[Interpreter] ABORT: Could not load model.")
            return

        # --- 1. SCAN DATASET ---
        top_activations = {i: [] for i in range(model.d_latent)}
        K_HEAP_SIZE = 10
        batch_size = 512
        processed = 0

        # Try OpenSearch scroll first, fallback to SQLite
        use_opensearch = False
        try:
            from search.client import is_available
            if is_available():
                from search.bulk_ops import scroll_documents_in_batches, count_documents
                total_docs = count_documents(run.dataset_id)
                use_opensearch = True
        except Exception:
            use_opensearch = False

        if not use_opensearch:
            total_docs = run.dataset.documents.filter(status='done').count()

        logger.info(f"[Interpreter] Scanning {total_docs} documents ({'OpenSearch' if use_opensearch else 'SQLite'})...")

        def _process_batch(batch_embeddings, batch_docs_info):
            """Process a batch: encode with SAE, track top activations."""
            nonlocal processed
            if not batch_embeddings:
                return

            try:
                X_batch = torch.tensor(batch_embeddings, dtype=torch.float32).to(device)
                X_batch = zscore_transform(X_batch, mean, std)

                with torch.no_grad():
                    feature_acts = model.encode(X_batch)

                feature_acts_cpu = feature_acts.cpu()

                mask = feature_acts_cpu > 0.001
                rows, cols = torch.where(mask)

                rows_np = rows.numpy()
                cols_np = cols.numpy()
                vals_np = feature_acts_cpu[mask].numpy()

                for r, c, v in zip(rows_np, cols_np, vals_np):
                    doc_info = batch_docs_info[r]
                    heap = top_activations[c]
                    if len(heap) < K_HEAP_SIZE:
                        heapq.heappush(heap, (v, doc_info[0], doc_info[1]))
                    elif v > heap[0][0]:
                        heapq.heapreplace(heap, (v, doc_info[0], doc_info[1]))

            except Exception as e:
                logger.error(f"[Interpreter] Batch error: {e}")

        if use_opensearch:
            for batch_data in scroll_documents_in_batches(run.dataset_id, batch_size=batch_size,
                                                           fields=['django_id', 'text', 'embedding']):
                if TASK_CONTROL.get(run_id) == 'STOP':
                    logger.info(f"[Interpreter] STOP signal received during scan. Aborting.")
                    if tid in TASK_PROGRESS:
                        TASK_PROGRESS[tid].update({'progress': 100, 'message': 'Paused by user.'})
                    return

                embeddings = [d['embedding'] for d in batch_data if d.get('embedding')]
                docs_info = [(d['django_id'], d.get('text', '')) for d in batch_data if d.get('embedding')]
                _process_batch(embeddings, docs_info)

                processed += len(batch_data)
                pct = int((processed / total_docs) * 50) if total_docs > 0 else 0
                if tid in TASK_PROGRESS:
                    TASK_PROGRESS[tid].update({'progress': pct, 'message': f'Scanning docs ({processed}/{total_docs})...'})
                if processed % 1000 == 0:
                    logger.info(f"[Interpreter] Processed {processed}/{total_docs} docs.")
        else:
            doc_qs = run.dataset.documents.filter(status='done').order_by('id')
            for offset in range(0, total_docs, batch_size):
                if TASK_CONTROL.get(run_id) == 'STOP':
                    logger.info(f"[Interpreter] STOP signal received during scan. Aborting.")
                    if tid in TASK_PROGRESS:
                        TASK_PROGRESS[tid].update({'progress': 100, 'message': 'Paused by user.'})
                    return

                batch_docs = list(doc_qs[offset:offset+batch_size])
                if not batch_docs: break

                embeddings = [d.embedding for d in batch_docs if d.embedding]
                docs_info = [(d.id, d.text) for d in batch_docs if d.embedding]
                _process_batch(embeddings, docs_info)

                processed += len(batch_docs)
                pct = int((processed / total_docs) * 50) if total_docs > 0 else 0
                if tid in TASK_PROGRESS:
                    TASK_PROGRESS[tid].update({'progress': pct, 'message': f'Scanning docs ({processed}/{total_docs})...'})
                if processed % 1000 == 0:
                    logger.info(f"[Interpreter] Processed {processed}/{total_docs} docs.")

        # --- 2. SELECT TOP FEATURES ---
        logger.info("\n[Interpreter] Selecting top features...")
        feature_stats = []
        for f_idx, heap in top_activations.items():
            if heap:
                max_act = max(item[0] for item in heap)
                feature_stats.append((max_act, f_idx))
        
        logger.info(f"[Interpreter] Found {len(feature_stats)} features with at least one activation.")
        
        if not feature_stats:
            logger.warning("[Interpreter] WARNING: No activations found > 0.001. Check normalization/model.")
            return

        feature_stats.sort(key=lambda x: x[0], reverse=True)
        logger.info(f"[Interpreter] Starting interpretation loop. Goal: {features_to_analyze} new interpretations.")

        # --- 3. LLM LOOP ---
        success_count = 0
        skipped_count = 0 
        
        for i, (max_act, f_idx) in enumerate(feature_stats):
            # CHECK FOR STOP SIGNAL
            if TASK_CONTROL.get(run_id) == 'STOP':
                logger.info(f"[Interpreter] STOP signal received. Pausing gracefully.")
                if tid in TASK_PROGRESS:
                    TASK_PROGRESS[tid].update({'progress': 100, 'message': 'Paused by user.'})
                break

            if success_count >= features_to_analyze:
                logger.info(f"[Interpreter] Reached target of {features_to_analyze} new interpretations. Stopping.")
                break
            
            # --- LOGICA SKIP FEATURE ---
            existing_feature = SAEFeature.objects.filter(run=run, feature_index=f_idx).first()
            if existing_feature and existing_feature.label:
                is_default_label = f"Feature {f_idx}" in existing_feature.label or "Feature #" in existing_feature.label or "Unknown" in existing_feature.label
                if not is_default_label:
                    logger.info(f"   -> Skipping Feat {f_idx} ({i+1}/{len(feature_stats)}): Already interpreted as '{existing_feature.label}'")
                    skipped_count += 1
                    continue
            # ---------------------------
            
            logger.info(f"\n--- Feature {f_idx} ({i+1}/{len(feature_stats)}) Max Act: {max_act:.2f} ---")
            
            # --- PROGRESS UPDATE (50-100%) ---
            # Use success_count to track progress towards the goal
            current_feat_pct = int((success_count / features_to_analyze) * 50)
            total_pct = 50 + current_feat_pct
            
            # Ensure we don't exceed 99% until done
            if total_pct >= 100: total_pct = 99
            
            if tid in TASK_PROGRESS:
                TASK_PROGRESS[tid].update({
                    'progress': total_pct, 
                    'message': f'Interpreting Feature {f_idx} (Success: {success_count}/{features_to_analyze})...'
                })
            # ---------------------------------
            
            pos_examples = sorted(top_activations[f_idx], key=lambda x: x[0], reverse=True)[:k_pos]
            
            # Recupero Negativi
            neg_examples = get_negative_examples(run, f_idx, k=k_neg, model=model, mean=mean, std=std)

            # Costruzione Prompt
            prompt_text = "Positive Examples (High Activation):\n"
            for val, did, txt in pos_examples:
                clean_txt = txt.replace('\n', ' ').strip()
                prompt_text += f"- [Act: {val:.2f}] {clean_txt}\n"
            
            prompt_text += "\nNegative Examples (Zero Activation):\n"
            if neg_examples:
                for doc in neg_examples:
                    clean_txt = doc['text'].replace('\n', ' ').strip()
                    prompt_text += f"- [Act: {doc['act']:.2f}] {clean_txt}\n"
            else:
                prompt_text += "- (No negative examples found, assuming random unrelated texts)\n"

            logger.info(f"   > Sending request to Ollama ({ollama_model})...")
            start_ts = time.time()
            result = get_ollama_response(prompt_text, custom_system_prompt, model=ollama_model, temperature=temp)
            
            if result:
                formatted_docs = [{'id': pid, 'act': float(pval), 'text': ptxt} for pval, pid, ptxt in pos_examples]
                
                feat_obj, _ = SAEFeature.objects.update_or_create(
                    run=run,
                    feature_index=f_idx,
                    defaults={
                        'label': result.get('label', 'Unknown'),
                        'description': result.get('description', ''),
                        'max_activation': float(max_act),
                        'example_docs': formatted_docs
                    }
                )
                
                interp = Interpretation.objects.create(
                    feature=feat_obj,
                    label=result.get('label', 'Unknown'),
                    description=result.get('description', ''),
                    llm_model=ollama_model,
                    system_prompt=custom_system_prompt,
                    temperature=temp,
                    evidence_docs={'positive': formatted_docs}
                )
                
                # LINK ACTIVE INTERPRETATION
                feat_obj.active_interpretation = interp
                feat_obj.save()

                # Phase 4: Index feature in OpenSearch
                try:
                    from search.client import is_available
                    if is_available():
                        from search.bulk_ops import update_feature
                        from search.indices import create_feature_index
                        create_feature_index(run.id)
                        update_feature(run.id, feat_obj.id, {
                            'django_id': feat_obj.id,
                            'feature_index': f_idx,
                            'label': feat_obj.label or '',
                            'description': feat_obj.description or '',
                            'max_activation': float(max_act),
                            'example_docs': formatted_docs,
                        })
                except Exception as os_err:
                    logger.warning(f"[OpenSearch] Failed to index feature: {os_err}")

                logger.info(f"   > SUCCESS: {result.get('label')} ({time.time()-start_ts:.1f}s)")
                success_count += 1
            else:
                logger.error(f"   > FAILED: No response from Ollama.")

        logger.info(f"\n[Interpreter] Batch Pipeline Complete. Interpreted: {success_count}. Skipped: {skipped_count}.")
        
    except Exception as e:
        logger.error(f"\n[Interpreter] CRITICAL PIPELINE ERROR: {e}", exc_info=True)
        # --- PROGRESS ERROR ---
        if 'tid' in locals():
            TASK_PROGRESS[tid] = {'progress': 100, 'message': f'Error: {str(e)}'}
        # ----------------------
    finally:
        # --- PROGRESS CLEANUP ---
        if 'tid' in locals() and tid in TASK_PROGRESS:
             # Keep it for a moment or mark done
             TASK_PROGRESS[tid] = {'progress': 100, 'message': 'Done.'}
        # ------------------------