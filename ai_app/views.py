from rest_framework import generics
from .models import ImageData, AnalysisResult
from .serializers import ImageDataSerializer, AnalysisResultSerializer
import torch
from torchvision import transforms
from PIL import Image
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import UploadedImage  
from .serializers import ImageSerializer,ImageUploadSerializer
import tensorflow as tf
from PIL import Image as PILImage
import json
import os
from torchvision.models import EfficientNet_B0_Weights
import torchvision.models as models
from django.conf import settings
from django.http import JsonResponse
from rest_framework.parsers import MultiPartParser, FormParser
from PIL import Image
import io
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from django.db.models import Count
from django.http import HttpResponse
import matplotlib.pyplot as plt
from io import BytesIO


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




class ImageRecognitionAPIView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            image_instance = serializer.save()
            prediction = self.predict_image(image_instance.image.path)
            image_instance.prediction = prediction
            image_instance.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def predict_image(self, image_path):
        # Example using TensorFlow
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        predicted_class = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0][0][1]
        return predicted_class

        # Example using PyTorch (Uncomment and modify as needed)
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model.eval()
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_image = PILImage.open(image_path)
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = model(input_batch)
        _, predicted = torch.max(output, 1)
        return predicted.item()  # Simplified for this example



def get_model():
    # Load EfficientNet model with up-to-date weights
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.eval()
    return model

class ImageUploadsView(APIView):
    def post(self, request, *args, **kwargs):
        # Retrieve the image file from the request
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'error': 'No image file provided'}, status=400)

        try:
            # Open and verify the image file
            image = PILImage.open(image_file)
            image.verify()  # Verify that it's an image file
            image = PILImage.open(image_file).convert('RGB')  # Reopen and convert to RGB after verification

        except UnidentifiedImageError:
            return JsonResponse({'error': 'Cannot identify image file'}, status=400)
        except IOError as e:
            return JsonResponse({'error': str(e)}, status=400)

        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image).unsqueeze(0)

        # Load the model
        model = self.get_model()

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = outputs.max(1)

        # Load class labels
        class_idx = self.get_labels()

        # Get prediction label
        predicted_index = str(predicted.item())
        prediction = class_idx.get(predicted_index, ['unknown'])[1]

        # Save the image and prediction to the database
        image_instance = UploadedImage(image=image_file, prediction=prediction)
        image_instance.save()

        return JsonResponse({'prediction': prediction})

    def get_model(self):
        # Load EfficientNet B0 model with up-to-date weights
        model = models.efficientnet_b0(weights='DEFAULT')
        model.eval()  # Set the model to evaluation mode
        return model

    def get_labels(self):
        # Load the ImageNet class index file
        with open('imagenet_class_index.json') as f:
            class_idx = json.load(f)
        return {str(k): v for k, v in class_idx.items()}





class GPTTextGeneration:
    def __init__(self):
        # Load the GPT-2 model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def generate_text(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class ImageToTextView(APIView):
    def post(self, request, *args, **kwargs):
        # Retrieve the image file from the request
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'error': 'No image file provided'}, status=400)

        try:
            # Open and verify the image file
            image = Image.open(image_file)
            image.verify()  # Verify that it's an image file
            image = Image.open(image_file).convert('RGB')  # Reopen and convert to RGB after verification

        except UnidentifiedImageError:
            return JsonResponse({'error': 'Cannot identify image file'}, status=400)
        except IOError as e:
            return JsonResponse({'error': str(e)}, status=400)

        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image).unsqueeze(0)

        # Predict using the model
        model = self.get_model()
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = outputs.max(1)

        # Load class labels
        class_idx = self.get_labels()
        predicted_index = str(predicted.item())
        prediction = class_idx.get(predicted_index, ['unknown'])[1]

        # Generate description using GPT-2
        gpt_generator = GPTTextGeneration()
        response_text = gpt_generator.generate_text(f"Describe an image with the label {prediction}.")

        # Save the image and prediction to the database
        image_instance = UploadedImage(image=image_file, prediction=prediction)
        image_instance.save()

        return JsonResponse({'prediction': prediction, 'description': response_text})

    def get_model(self):
        # Load EfficientNet B0 model with up-to-date weights
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.eval()  # Set the model to evaluation mode
        return model

    def get_labels(self):
        # Load the ImageNet class index file
        with open('imagenet_class_index.json') as f:
            class_idx = json.load(f)
        return {str(k): v for k, v in class_idx.items()}  # Ensure keys are strings







def predictions_chart(request):
    predictions = UploadedImage.objects.values('prediction').annotate(count=Count('prediction')).order_by('-count')

    labels = [p['prediction'] for p in predictions]
    sizes = [p['count'] for p in predictions]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return HttpResponse(buffer, content_type='image/png')
