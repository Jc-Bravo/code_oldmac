import datetime

from django.db import models
from django.utils import timezone
from django.contrib import admin

# Create your models here.
class Question(models.Model):
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')
    def __str__(self):
        return self.question_text
    @admin.display(
        boolean=True,
        ordering='pub_date',
        description='Published recently?',
    )
    def was_published_recently(self):
        now = timezone.now()
        return now - datetime.timedelta(days=1) <= self.pub_date <= now

class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)
    def __str__(self):
        return self.choice_text


class MachineLearningManager(models.Manager):
    def create_model(self, model_name, sponsor_name, sponsor_time, complete_time, train_status, train_duration,
                     train_console):
        mlm = self.create(model_name=model_name, sponsor_name=sponsor_name, sponsor_time=sponsor_time,
                          complete_time=complete_time, train_status=train_status, train_duration=train_duration,
                          train_console=train_console)
        # do something with the model
        return mlm


class MachineLearningModel(models.Model):
    model_name = models.CharField(max_length=20)
    sponsor_name = models.CharField(max_length=20)
    sponsor_time = models.DateTimeField('when training begins')
    complete_time = models.DateTimeField('when training completes', default=datetime.datetime.now())
    train_status = models.BooleanField('whether training completes')
    train_duration = models.IntegerField('how long the training lasts')
    train_console = models.CharField('Console', max_length=2000)

    objects = MachineLearningManager()

    def __str__(self):
        return self.model_name