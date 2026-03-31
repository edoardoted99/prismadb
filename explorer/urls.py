from django.urls import path

from . import views

app_name = 'explorer'

urlpatterns = [
    path('', views.index, name='index'),
    path('interpret/', views.start_interpretation, name='start_interpretation'),

    # Lista + Search
    path('run/<int:run_id>/', views.feature_list, name='feature_list'),

    # Actions
    path('run/<int:run_id>/stats/', views.calculate_stats, name='calculate_stats'),
    path('run/<int:run_id>/export_stats/', views.export_feature_statistics, name='export_feature_statistics'),
    path('run/<int:run_id>/export_docs/', views.export_document_activations, name='export_document_activations'),
    path('run/<int:run_id>/download_weights/', views.download_sae_weights, name='download_sae_weights'),

    # Dettaglio Feature
    path('run/<int:run_id>/feature/<int:feature_index>/', views.feature_detail, name='feature_detail'),

    # --- NUOVA ROTTA: Re-interpretazione singola (usata dal Modal) ---
    path('run/<int:run_id>/feature/<int:feature_index>/reinterpret/', views.reinterpret_feature, name='reinterpret_feature'),

    #path('run/<int:run_id>/graph/', views.knowledge_graph, name='knowledge_graph'),
    #path('run/<int:run_id>/graph/data/', views.knowledge_graph_data, name='knowledge_graph_data'),
    path('run/<int:run_id>/families/', views.family_list, name='family_list'),
    path('run/<int:run_id>/families/build/', views.build_families, name='build_families'),
    path('analyzer/', views.document_analyzer, name='document_analyzer'),
    path('inference/', views.inference_view, name='inference'),

    # System Status
    path('system_status/', views.system_status, name='system_status'),
    path('system_status/logs/', views.get_logs, name='get_logs'),
    path('system_status/threads/', views.get_threads, name='get_threads'),
    path('system_status/stats/', views.get_system_stats, name='get_system_stats'),
    path('system_status/services/', views.get_services_status, name='get_services_status'),
    path('system_status/ollama_url/', views.update_ollama_url, name='update_ollama_url'),
    path('system_status/kill/<int:ident>/', views.kill_thread, name='kill_thread'),
    path('system_status/stop/<int:run_id>/', views.stop_interpretation, name='stop_interpretation'),

    # Debug
    path('debug/download_db/', views.download_db, name='download_db'),
]
