# Generated by Django 2.0.13 on 2020-01-22 05:32

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Param',
            fields=[
                ('param_id', models.IntegerField(primary_key=True, serialize=False, verbose_name='Номер параметра')),
                ('param_name', models.CharField(max_length=20, unique=True, verbose_name='Параметр')),
                ('param_value', models.DecimalField(decimal_places=7, max_digits=11, verbose_name='Значение параметра')),
                ('problem', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='problem', to='blog.Post')),
            ],
        ),
    ]
