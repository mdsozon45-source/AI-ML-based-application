from django.urls import path
from .views import ImageUploadView, ImageAnalysisView

urlpatterns = [
    path('upload/', ImageUploadView.as_view(), name='image-upload'),
    path('analysis/<int:pk>/', ImageAnalysisView.as_view(), name='image-analysis'),
]
