# Generated by Django 2.0.13 on 2020-02-14 10:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0003_param_param_ed'),
    ]

    operations = [
        migrations.AlterField(
            model_name='param',
            name='param_ed',
            field=models.CharField(max_length=20, null=True, verbose_name='Единица измерения'),
        ),
    ]