from django.db import models

class FaceImage(models.Model):
    image = models.ImageField(upload_to='faces/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    recognized = models.BooleanField(default=False)

    def __str__(self):
        return f"Image {self.id} - Recognized: {self.recognized}"

class RecognizedFace(models.Model):
    face_image = models.ForeignKey(FaceImage, on_delete=models.CASCADE, related_name='recognized_faces')
    name = models.CharField(max_length=255, blank=True, null=True)
    confidence = models.FloatField(default=0.0)

    def __str__(self):
        return f"Face: {self.name} - Confidence: {self.confidence}"




class KnownFace(models.Model):
    name = models.CharField(max_length=255)
    encoding = models.JSONField()  # Store the face encoding as a JSON array
    image = models.ImageField(upload_to='known_faces/')  # Optional: To store the image of the known face

    def __str__(self):
        return self.name