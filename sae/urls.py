# sae/urls.py
from django.urls import path

from . import views

app_name = 'sae'

urlpatterns = [
    path('', views.run_list, name='run_list'),
    path('create/', views.create_run, name='create_run'),
    path('<int:pk>/', views.run_detail, name='run_detail'),
    path('<int:pk>/start/', views.start_run, name='start_run'),
]
