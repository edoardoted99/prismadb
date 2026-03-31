from django.urls import include, path
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularSwaggerView,
)
from rest_framework.routers import DefaultRouter

from . import views

app_name = "api"

router = DefaultRouter()
router.register(r"datasets", views.DatasetViewSet)
router.register(r"runs", views.SAERunViewSet)

urlpatterns = [
    path("", include(router.urls)),
    path("inference/", views.inference, name="inference"),
    path("status/", views.system_status, name="system-status"),
    path("search/bm25/", views.search_bm25, name="search-bm25"),
    path("search/semantic/", views.search_semantic, name="search-semantic"),
    path("search/hybrid/", views.search_hybrid, name="search-hybrid"),
    # OpenAPI
    path("schema/", SpectacularAPIView.as_view(), name="schema"),
    path("docs/", SpectacularSwaggerView.as_view(url_name="api:schema"), name="swagger"),
]
