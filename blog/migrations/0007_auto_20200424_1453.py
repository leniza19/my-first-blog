# Generated by Django 2.0.13 on 2020-04-24 11:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0006_post_problem_id'),
    ]

    operations = [
        migrations.AlterField(
            model_name='param',
            name='param_name',
            field=models.CharField(max_length=20, verbose_name='Параметр'),
        ),
    ]
