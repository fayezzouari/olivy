# Generated by Django 4.2.3 on 2023-12-11 17:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('disease_detection', '0003_users_data_delete_camera'),
    ]

    operations = [
        migrations.AlterField(
            model_name='users_data',
            name='user_email',
            field=models.EmailField(max_length=254, null=True),
        ),
        migrations.AlterField(
            model_name='users_data',
            name='user_name',
            field=models.CharField(max_length=255, null=True),
        ),
    ]
