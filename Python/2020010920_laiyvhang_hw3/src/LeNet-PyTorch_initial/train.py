#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/8/28 11:04
# @File     : train.py

"""
import argparse
import copy

import torch
import torchvision

from models.lenet import LeNet
from utils import pre_process

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms
import cv2

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
        print_error = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if (predicted == labels).sum().item() <= 245 & (predicted == labels).sum().item() != 16:
                output = outputs[0].argmax()
                print(f'inference label is: {output}')
                img = images[0]
                img = img.cpu().numpy()
                img = np.transpose(img, (1, 2, 0))
                plt.imshow(img)
                plt.show()
                cv2.waitKey(0)

        print('Test Accuracy of the model is: {} %'.format(100 * correct / total))


def save_model(model, save_path='lenet.pth'):
    ckpt_dict = {
        'state_dict': model.state_dict()
    }
    torch.save(ckpt_dict, save_path)


def train(epochs, batch_size, learning_rate, num_classes):

    # fetch data
    train_loader, test_loader = get_data_loader(batch_size)

    # Loss and optimizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = LeNet(num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # prepare painting (step [1/3])
    loss_ls = []
    average_loss_per_epoch = 0

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

            # record loss (step [2/3])
            average_loss_per_epoch += loss.item()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, total_step, loss.item()))

        # calculate average loss
        average_loss_per_epoch = average_loss_per_epoch / float(total_step)
        loss_ls.append(copy.deepcopy(average_loss_per_epoch))
        average_loss_per_epoch = 0

        # evaluate after epoch train
        evaluate(model, test_loader, device)

    # paint loss_log (step [3/3])
    plt.figure(figsize=(5, 2.7), layout='constrained')
    plt.plot(range(1, 11), loss_ls, label='loss')
    plt.xlabel('epoch')
    plt.ylabel('average loss')
    plt.title("Loss per epoch")
    plt.legend()
    plt.show()

    # save the trained model
    save_model(model, save_path='lenet.pth')
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_classes', type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train(args.epochs, args.batch_size, args.lr, args.num_classes)


