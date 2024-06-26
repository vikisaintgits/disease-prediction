# Generated by Django 4.0.1 on 2022-03-08 12:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('hospital', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='appointment',
            name='App_files',
            field=models.FileField(default=1, max_length=1000, upload_to='appfiles/', verbose_name='app_files'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='appointment',
            name='conf_date',
            field=models.CharField(default=1, max_length=100, verbose_name='conf_date'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='user',
            name='User_files',
            field=models.FileField(default=1, max_length=1000, upload_to='userfiles/', verbose_name='User_files'),
            preserve_default=False,
        ),
    ]
