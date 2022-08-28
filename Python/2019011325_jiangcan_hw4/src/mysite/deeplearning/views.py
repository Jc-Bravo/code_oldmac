import datetime
import threading

from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.utils import timezone
from django.views import generic

from .models import Question, Choice, MachineLearningModel, MachineLearningManager
from datetime import datetime

import torch
import torchvision

from models.lenet import LeNet
from utils import pre_process

import numpy as np

from torchvision import transforms

def get_data_loader(batch_size):
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='data/',
                                               train=True,
                                               transform=pre_process.data_augment_transform(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='data/',
                                              train=False,
                                              transform=pre_process.normal_transform())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_loader, test_loader


def evaluate(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model is: {} %'.format(100 * correct / total))


def save_model(model, save_path='lenet.pth'):
    ckpt_dict = {
        'state_dict': model.state_dict()
    }
    torch.save(ckpt_dict, save_path)


def train(epochs, batch_size, learning_rate, num_classes, Model):
    # fetch data
    train_loader, test_loader = get_data_loader(batch_size)

    # Loss and optimizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = LeNet(num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # fetch console
    console_in_django = ''

    # start train
    total_step = len(train_loader)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):

            # get image and label
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, total_step, loss.item()))
                # write in console
                data = loss.item()
                data = format(data, '.4f')
                data = str(data)
                Model.train_console = Model.train_console + '/' + data
                Model.save()
                # may cut off a null str

        # evaluate after epoch train
        evaluate(model, test_loader, device)

    # save the trained model and console
    save_model(model, save_path='lenet.pth')
    return model


def index(request):
    # TODO 添加多线程算法 muti-threading
    
    if request.method == 'POST':
        # thread2 = threading.Thread(target=jump_result(request))
        # thread1 = threading.Thread(target=begin_learning(request))
        # x = thread2.start()
        # thread1.join()
        # return x
        print("I've receive the command.")
        # create your model and save it now!
        model_name = request.POST.get('model_name')
        sponsor_name = request.POST.get('sponsor_name')
        sponsor_time = datetime.now()
        complete_time = datetime.now()
        train_status = False
        train_duration = 0
        train_console = ''
        # remember to update the console
        Model = MachineLearningModel.objects.create_model(model_name, sponsor_name, sponsor_time, complete_time,
                                                          train_status, train_duration, train_console)
        train(10, 256, 0.001, 10, Model)
        Model.complete_time = datetime.now()
        Model.train_status = True
        Model.save()
        return HttpResponseRedirect(reverse('deeplearning:results'))
    else:
        # if there be a multi-thread operation, it should be here
        print("it is get.")
        # return HttpResponseRedirect(reverse('deeplearning:results'))

        return render(request, 'deeplearning/index.html')

# begin train
# def begin_learning(request):
#     print("begin training!")
#     # create your model and save it now!
#     model_name = request.POST.get('model_name')
#     sponsor_name = request.POST.get('sponsor_name')
#     sponsor_time = datetime.now()
#     complete_time = datetime.now()
#     train_status = False
#     train_duration = 0
#     train_console = ''
#     # remember to update the console
#     Model = MachineLearningModel.objects.create_model(model_name, sponsor_name, sponsor_time, complete_time,
#                                                         train_status, train_duration, train_console)
#     train(10, 256, 0.001, 10, Model)
#     Model.complete_time = datetime.now()
#     Model.train_status = True
#     Model.save()
#     return HttpResponseRedirect(reverse('deeplearning:results'))

# jump to the result
# def jump_result(request):
#     print("success jump!")
#     return HttpResponseRedirect(reverse('deeplearning:results'))

def detail(request, project_id):
    # may be some try and except module?
    train_project = get_object_or_404(MachineLearningModel, pk=project_id)
    try:
        if not train_project.train_status:
            train_project.train_duration = (datetime.now() - train_project.sponsor_time).seconds
            train_project.save()
        else:
            train_project.train_duration = (train_project.complete_time - train_project.sponsor_time).seconds
            train_project.save()
    except KeyError:
        train_project.sponsor_time = timezone.now()
        train_project.train_duration = 0
        train_project.save()
    train_console = train_project.train_console
    train_console = train_console.split('/')
    train_console = train_console[1:]
    train_console_length = len(train_console)
    train_completed = train_project.complete_time
    context = {
        'train_project': train_project,
        'train_completed': train_completed,
        'train_console': train_console,
        'train_console_length': train_console_length,
    }
    return render(request, 'deeplearning/detail.html', context)


def results(request):
    project_list = MachineLearningModel.objects.all()
    project_list = list(project_list)
    context = {'project_list': project_list}
    return render(request, 'deeplearning/results.html', context)


def home(request):
    return render(request, 'deeplearning/home.html')
