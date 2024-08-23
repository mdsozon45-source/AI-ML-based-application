from django.urls import path
from .views import ImageUploadView,predictions_chart,ImageToTextView,ImageUploadsView, ImageAnalysisView,ImageRecognitionAPIView

urlpatterns = [
    path('upload/', ImageUploadView.as_view(), name='image-upload'),
    path('analysis/<int:pk>/', ImageAnalysisView.as_view(), name='image-analysis'),
    path('analyze/', ImageRecognitionAPIView.as_view(), name='image_recognition_api'),
    path('uploads/', ImageUploadsView.as_view(), name='image-uploads'),
    path('predictions-chart/', predictions_chart, name='predictions_chart'),
    path('image-to-text/', ImageToTextView.as_view(), name='image-to-text-upload'),
]
