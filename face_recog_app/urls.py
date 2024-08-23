from django.urls import path
from .views import FaceRecognitionAPI,AddKnownFaceAPI

urlpatterns = [
    path('face-recognition/', FaceRecognitionAPI.as_view(), name='face_recognition_api'),
    path('add-known-face/', AddKnownFaceAPI.as_view(), name='add-known-face'),
]
