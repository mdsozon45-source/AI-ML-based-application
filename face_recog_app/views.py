from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import face_recognition
import numpy as np
from .models import FaceImage, RecognizedFace,KnownFace
from .serializers import FaceImageSerializer,KnownFaceSerializer




class AddKnownFaceAPI(APIView):
    def post(self, request, *args, **kwargs):
        image_file = request.FILES.get('image')
        name = request.data.get('name')

        if not image_file or not name:
            return Response({'error': 'Name and image file are required'}, status=status.HTTP_400_BAD_REQUEST)

        # Load the image
        image = face_recognition.load_image_file(image_file)

        # Get the face encodings
        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:
            # Assume only one face per image
            face_encoding = face_encodings[0]
            face_encoding_list = face_encoding.tolist()

            # Save the known face to the database
            known_face = KnownFace(name=name, encoding=face_encoding_list)
            known_face.image.save(image_file.name, image_file)  # Save the image
            known_face.save()

            return Response({'message': 'Face registered successfully'}, status=status.HTTP_201_CREATED)
        else:
            return Response({'error': 'No faces found in the image'}, status=status.HTTP_400_BAD_REQUEST)




class FaceRecognitionAPI(APIView):
    def post(self, request, *args, **kwargs):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({'error': 'No image file provided'}, status=status.HTTP_400_BAD_REQUEST)

        # Save the image
        face_image = FaceImage(image=image_file)
        face_image.save()

        try:
            # Load the image with face_recognition
            image_path = face_image.image.path
            image = face_recognition.load_image_file(image_path)

            # Detect faces
            face_locations = face_recognition.face_locations(image)

            if not face_locations:
                return Response({'message': 'No faces found in the image'}, status=status.HTTP_200_OK)

            # Recognize faces
            face_encodings = face_recognition.face_encodings(image, face_locations)

            # Load known faces from the database
            known_faces = KnownFace.objects.all()
            known_face_encodings = [np.array(face.encoding) for face in known_faces]
            known_face_names = [face.name for face in known_faces]

            if not known_face_encodings:
                return Response({'error': 'No known faces available for comparison'}, status=status.HTTP_400_BAD_REQUEST)

            recognized_faces = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                    # Check if this face has already been recognized
                    if not any(rf['name'] == name and abs(rf['confidence'] - confidence) < 0.1 for rf in recognized_faces):
                        recognized_faces.append({
                            'name': name,
                            'confidence': confidence
                        })

                        # Save the recognized face information
                        recognized_face = RecognizedFace(
                            face_image=face_image,
                            name=name,
                            confidence=confidence
                        )
                        recognized_face.save()
                        face_image.recognized = True
                        face_image.save()
                else:
                    name = "Unknown"
                    confidence = 0.0

                

            

            return Response({'message': 'Faces recognized', 'data': FaceImageSerializer(face_image).data}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)