
from django.core.management.base import BaseCommand
from ai_app.models import UploadedImage  
from django.db.models import Count
class Command(BaseCommand):
    help = 'Analyze image data and generate insights'

    def handle(self, *args, **kwargs):
        total_images = UploadedImage.objects.count()
        distinct_predictions = UploadedImage.objects.values('prediction').distinct().count()
        most_common_prediction = UploadedImage.objects.values('prediction').annotate(count=Count('prediction')).order_by('-count').first()

        self.stdout.write(f'Total images: {total_images}')
        self.stdout.write(f'Distinct predictions: {distinct_predictions}')
        self.stdout.write(f'Most common prediction: {most_common_prediction["prediction"] if most_common_prediction else "None"}')