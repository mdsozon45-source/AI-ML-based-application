# Generated by Django 4.2.15 on 2024-08-23 20:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("face_recog_app", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="KnownFace",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=255)),
                ("encoding", models.JSONField()),
                ("image", models.ImageField(upload_to="known_faces/")),
            ],
        ),
    ]