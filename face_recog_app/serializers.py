from rest_framework import serializers
from .models import FaceImage, RecognizedFace,KnownFace

class RecognizedFaceSerializer(serializers.ModelSerializer):
    class Meta:
        model = RecognizedFace
        fields = '__all__'

class FaceImageSerializer(serializers.ModelSerializer):
    recognized_faces = RecognizedFaceSerializer(many=True, read_only=True)

    class Meta:
        model = FaceImage
        fields = '__all__'




class KnownFaceSerializer(serializers.ModelSerializer):
    class Meta:
        model = KnownFace
        fields = ['name', 'encoding', 'image']
        read_only_fields = ['encoding']