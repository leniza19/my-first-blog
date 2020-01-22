from django.conf import settings
from django.db import models
from django.utils import timezone


class Post(models.Model):
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    text = models.TextField()
    created_date = models.DateTimeField(default=timezone.now)
    published_date = models.DateTimeField(blank=True, null=True)

    def publish(self):
        self.published_date = timezone.now()
        self.save()

    def __str__(self):
        return self.title


class Param(models.Model):
    param_id = models.IntegerField(primary_key=True, verbose_name='Номер параметра')
    param_name = models.CharField(max_length=20, unique=True, verbose_name='Параметр')
    param_value = models.DecimalField(max_digits=11, decimal_places=7, verbose_name='Значение параметра')
    problem = models.ForeignKey(Post, null=False, on_delete=models.CASCADE, related_name='problem')

    def __str__(self):
        return self.param_name
