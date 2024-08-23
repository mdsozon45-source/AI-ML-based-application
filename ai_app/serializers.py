from rest_framework import serializers
from .models import ImageData, AnalysisResult

class ImageDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageData
        fields = ['id', 'image', 'uploaded_at', 'processed']

class AnalysisResultSerializer(serializers.ModelSerializer):
    image = ImageDataSerializer()

    class Meta:
        model = AnalysisResult
        fields = ['id', 'image', 'result', 'created_at']
