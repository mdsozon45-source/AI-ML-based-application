# Generated by Django 4.2.15 on 2024-08-23 18:25

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("ai_app", "0002_image"),
    ]

    operations = [
        migrations.RenameModel(
            old_name="Image",
            new_name="UploadedImage",
        ),
    ]
