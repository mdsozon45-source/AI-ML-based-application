from django.contrib import admin
from .models import ImageData, AnalysisResult,UploadedImage

admin.site.register(ImageData)
admin.site.register(AnalysisResult)
admin.site.register(UploadedImage)