from django.urls import path

from . import views

app_name = "embeddings"


urlpatterns = [
    path("", views.dataset_list, name="dataset_list"),
    path("upload/", views.upload_dataset, name="upload_dataset"),
    path("<int:pk>/", views.dataset_detail, name="dataset_detail"),
    path("<int:pk>/start-generation/", views.start_generation, name="start_generation"),
    path("<int:pk>/delete/", views.delete_dataset, name="delete_dataset"), # <--- NUOVA
    path("document/<int:doc_id>/", views.document_detail, name="document_detail"),  # <--

]
