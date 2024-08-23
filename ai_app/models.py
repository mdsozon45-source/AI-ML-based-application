from django.db import models




class ImageData(models.Model):
    image = models.ImageField(upload_to='images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)

class AnalysisResult(models.Model):
    image = models.ForeignKey(ImageData, on_delete=models.CASCADE)
    result = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)


class UploadedImage(models.Model):
    image = models.ImageField(upload_to='images/')
    prediction = models.CharField(max_length=255, blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.image.name