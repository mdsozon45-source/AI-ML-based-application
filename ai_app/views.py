from rest_framework import generics
from .models import ImageData, AnalysisResult
from .serializers import ImageDataSerializer, AnalysisResultSerializer
import torch
from torchvision import transforms
from PIL import Image

# class ImageUploadView(generics.CreateAPIView):
#     queryset = ImageData.objects.all()
#     serializer_class = ImageDataSerializer

#     def perform_create(self, serializer):
#         instance = serializer.save()
#         # Placeholder for AI model processing
#         result = "Processed result goes here"
#         AnalysisResult.objects.create(image=instance, result=result)



class ImageUploadView(generics.CreateAPIView):
    queryset = ImageData.objects.all()
    serializer_class = ImageDataSerializer

    def perform_create(self, serializer):
        instance = serializer.save()
        image_path = instance.image.path

       
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model.eval()

        input_image = Image.open(image_path)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        with torch.no_grad():
            output = model(input_batch)
        
        result = torch.nn.functional.softmax(output[0], dim=0)
        AnalysisResult.objects.create(image=instance, result=str(result))

class ImageAnalysisView(generics.RetrieveAPIView):
    queryset = AnalysisResult.objects.all()
    serializer_class = AnalysisResultSerializer
