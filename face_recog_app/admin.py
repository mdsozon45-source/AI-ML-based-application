from django.contrib import admin
from .models import FaceImage,RecognizedFace,KnownFace

admin.site.register(FaceImage)
admin.site.register(RecognizedFace)
admin.site.register(KnownFace)
