# DATS 6203 Machine Learning II
# Final Project
# Team: David Robison, Nathan Zencey, and Sadaf Asrar
# Code by: Sadaf Asrar

## Importing relevant packages

# Transform,Dataloader for pytorch
from __future__ import print_function, division
import torch
import torchvision
import pandas as pd

import os
from glob import glob
import time

# visualize images
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Preprocessing
import cv2
from skimage import io, transform
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode


# ### Define LeNet5 Network for training

##### Using SGD as optimizer
start = time.time()


'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # starting at 224 with max pooling kernel of 2 and stride of 2 results in 53x53
        # and 16 channels by the time we arrive at fully connected layer

        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        return out


net = LeNet().cuda()

# In[58]:


import torch.optim as optim

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



t_running_loss=[]
t_accuracy=[]

lab_cm=[]
pred_cm=[]


for epoch in range(10):  # loop over the dataset multiple times
    # model.train()
    # call ceiling on number of mini batches
    running_loss = 0.0
    for i, data in enumerate(trainDataLoader, 0):
        # get the inputs
        inputs = data['image']
        inputs = inputs.float()
        labels = data['label'].float()

        # wrap them in Variable
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 1000 == 0:  # print every 1000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))  # print running loss divided by total number of mini-batches
            #             running_loss = 0.0 shows running loss at the end of each epoch rather than every minibatch
            # model.eval() - also negates val is for tuning of hyperparameter and know when to stop training
    print(running_loss/1000)
    r_l_k=(running_loss/1000)
    t_running_loss.insert(0, r_l_k)
    t_running_loss_2 = t_running_loss[::-1]

##Testing

    correct = 0
    total = 0
    for data in testDataLoader:
        images = data['image'].float()
        labels = data['label'].type(torch.FloatTensor)
        outputs = net(Variable(images).cuda())
        predicted = outputs > 0.5
        predicted = predicted.data.cpu().numpy()

        total += labels.size(0)
        correct += (predicted.reshape(16) == labels.numpy()).sum()


    print('Accuracy of the network on the 4000 test images: %d %%' % (
        100 * correct / total))

    accuracy=(100 * correct / total)
    t_accuracy.insert(0, accuracy)
    t_accuracy_2 = t_accuracy[::-1]

    for l in labels:
        lab_cm.append(l)

    for p in predicted:
        pred_cm.append(p)

y_test.sum() / len(y_test)

ep=np.arange(0, 10, 1)

print(ep)
print(t_running_loss_2)
print(t_accuracy_2)

print('lab')
print(lab_cm)
print('PRED')
print(pred_cm)

print ('It took', time.time() - start, 'seconds.')


plt.interactive(False)


#Plot of training loss

plt.plot(ep, t_running_loss_2)
plt.ylabel('Training Loss Values')
plt.xlabel('Number of Epochs')

plt.show()

#Plot of accuracy
plt.plot(ep, t_accuracy_2)
plt.ylabel('Accuracy')
plt.xlabel('Number of Epochs')

plt.show()

classes=(1.0, 0.0)

cm_sgd=confusion_matrix(lab_cm, pred_cm)
print(cm_sgd)
#print(lab_cm, pred_cm)

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#plt.figure()
plot_confusion_matrix(cm_sgd, classes=classes, normalize=True,
                      title='Normalized confusion matrix: SGD')
plt.show()

from sklearn.metrics import f1_score


print('F1 score')
print(f1_score(lab_cm, pred_cm, average='weighted'))

####

##### Using Adam as optimizer

start = time.time()


'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # starting at 224 with max pooling kernel of 2 and stride of 2 results in 53x53
        # and 16 channels by the time we arrive at fully connected layer

        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        return out


net = LeNet().cuda()

# In[58]:


import torch.optim as optim

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)



t_running_loss_adam=[]
t_accuracy_adam=[]
for epoch in range(10):  # loop over the dataset multiple times
    # model.train()
    # call ceiling on number of mini batches
    running_loss = 0.0
    for i, data in enumerate(trainDataLoader, 0):
        # get the inputs
        inputs = data['image']
        inputs = inputs.float()
        labels = data['label'].float()

        # wrap them in Variable
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 1000 == 0:  # print every 1000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))  # print running loss divided by total number of mini-batches
            #             running_loss = 0.0 shows running loss at the end of each epoch rather than every minibatch
            # model.eval() - also negates val is for tuning of hyperparameter and know when to stop training
    print(running_loss/1000)
    r_l_k=(running_loss/1000)
    t_running_loss_adam.insert(0, r_l_k)
    t_running_loss_adam_2 = t_running_loss_adam[::-1]

##Testing

    correct = 0
    total = 0
    for data in testDataLoader:
        images = data['image'].float()
        labels = data['label'].type(torch.FloatTensor)
        outputs = net(Variable(images).cuda())
        predicted = outputs > 0.5
        predicted = predicted.data.cpu().numpy()

        total += labels.size(0)
        correct += (predicted.reshape(16) == labels.numpy()).sum()


    print('Accuracy of the network on the 4000 test images: %d %%' % (
        100 * correct / total))

    accuracy=(100 * correct / total)
    t_accuracy_adam.insert(0, accuracy)
    t_accuracy_adam_2 = t_accuracy_adam[::-1]

    for l in labels:
        lab_cm.append(l)

    for p in predicted:
        pred_cm.append(p)



y_test.sum() / len(y_test)

ep=np.arange(0, 10, 1)

print(ep)
print(t_running_loss_adam_2)
print(t_accuracy_adam_2)

print('lab')
print(lab_cm)
print('PRED')
print(pred_cm)

print ('It took', time.time() - start, 'seconds.')


plt.interactive(False)


#Plotting training loss and accuracy

plt.plot(ep, t_running_loss_adam_2)
plt.ylabel('Training Loss Values')
plt.xlabel('Number of Epochs')

plt.show()


plt.plot(ep, t_accuracy_adam_2)
plt.ylabel('Accuracy')
plt.xlabel('Number of Epochs')

plt.show()

cm_sgd=confusion_matrix(lab_cm, pred_cm)
print(cm_sgd)
#print(lab_cm, pred_cm)

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#plt.figure()
plot_confusion_matrix(cm_sgd, classes=classes, normalize=True,
                      title='Normalized confusion matrix: Adam')
plt.show()

from sklearn.metrics import f1_score


print('F1 score')
print(f1_score(lab_cm, pred_cm, average='weighted'))



##### Using Adagrad as optimizer

start = time.time()


'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # starting at 224 with max pooling kernel of 2 and stride of 2 results in 53x53
        # and 16 channels by the time we arrive at fully connected layer

        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        return out


net = LeNet().cuda()



import torch.optim as optim

criterion = nn.BCELoss()
optimizer = optim.Adagrad(net.parameters(), lr=0.001)


t_running_loss_adagrad=[]
t_accuracy_adagrad=[]
for epoch in range(10):  # loop over the dataset multiple times
    # model.train()
    # call ceiling on number of mini batches
    running_loss = 0.0
    for i, data in enumerate(trainDataLoader, 0):
        # get the inputs
        inputs = data['image']
        inputs = inputs.float()
        labels = data['label'].float()

        # wrap them in Variable
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 1000 == 0:  # print every 1000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))  # print running loss divided by total number of mini-batches
            #             running_loss = 0.0 shows running loss at the end of each epoch rather than every minibatch
            # model.eval() - also negates val is for tuning of hyperparameter and know when to stop training
    print(running_loss/1000)
    r_l_k=(running_loss/1000)
    t_running_loss_adagrad.insert(0, r_l_k)
    t_running_loss_adagrad_2 = t_running_loss_adagrad[::-1]

##Testing

    correct = 0
    total = 0
    for data in testDataLoader:
        images = data['image'].float()
        labels = data['label'].type(torch.FloatTensor)
        outputs = net(Variable(images).cuda())
        predicted = outputs > 0.5
        predicted = predicted.data.cpu().numpy()

        total += labels.size(0)
        correct += (predicted.reshape(16) == labels.numpy()).sum()


    print('Accuracy of the network on the 4000 test images: %d %%' % (
        100 * correct / total))

    accuracy=(100 * correct / total)
    t_accuracy_adagrad.insert(0, accuracy)
    t_accuracy_adagrad_2 = t_accuracy_adagrad[::-1]

    for l in labels:
        lab_cm.append(l)

    for p in predicted:
        pred_cm.append(p)

y_test.sum() / len(y_test)

ep=np.arange(0, 10, 1)

print(ep)
print(t_running_loss_adagrad_2)
print(t_accuracy_adagrad_2)

print('lab')
print(lab_cm)
print('PRED')
print(pred_cm)

print ('It took', time.time() - start, 'seconds.')


plt.interactive(False)


#Plotting training loss and accuracy

plt.plot(ep, t_running_loss_adagrad_2)
plt.ylabel('Training Loss Values')
plt.xlabel('Number of Epochs')

plt.show()


plt.plot(ep, t_accuracy_adagrad_2)
plt.ylabel('Accuracy')
plt.xlabel('Number of Epochs')

plt.show()

cm_sgd=confusion_matrix(lab_cm, pred_cm)
print(cm_sgd)
#print(lab_cm, pred_cm)

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#plt.figure()
plot_confusion_matrix(cm_sgd, classes=classes, normalize=True,
                      title='Normalized confusion matrix: Adagrad')
plt.show()

from sklearn.metrics import f1_score


print('F1 score')
print(f1_score(lab_cm, pred_cm, average='weighted'))



##########

#Plotting training loss for all three optimizers

plt.plot(ep, t_running_loss_2)
plt.plot(ep, t_running_loss_adam_2)
plt.plot(ep, t_running_loss_adagrad_2)
plt.ylabel('Training Loss Values')
plt.xlabel('Number of Epochs')
plt.legend(['SGD', 'Adam', 'Adagrad'],loc='center left')
plt.title('Training Loss by Optimizer')

plt.show()


#Plotting accuracy for all three optimizers


plt.plot(ep, t_accuracy_2)
plt.plot(ep, t_accuracy_adam_2)
plt.plot(ep, t_accuracy_adagrad_2)
plt.ylabel('Accuracy')
plt.xlabel('Number of Epochs')
plt.legend(['SGD', 'Adam', 'Adagrad'],loc='center left')
plt.title('Accuracy by Optimizer')

plt.show()
