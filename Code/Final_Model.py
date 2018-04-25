'''This script runs a final 'best model' after previous experiments with kernel 
size, transfer functions, and optimization algorithms.This script should be run with the 
data in the working directory'''


##### Import packages #########
import os
from glob import glob

#visualize images
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#Preprocessing
import cv2
from skimage import io, transform
import numpy as np

#Transform,Dataloader for pytorch
from __future__ import print_function, division
import torch
import torchvision
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


# ### Define LeNet5 with Kernel Size 15 x 15 SGD and ReLU
'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LeNet(nn.Module):
    def __init__(self,):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 15)
        self.conv2 = nn.Conv2d(6, 16, 15)
                
        self.fc1   = nn.Linear(16*45*45, 120)
        print(self.fc1)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 1)
        
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


import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

net = LeNet().cuda()    
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


losses = []
testLosses = []
testAccuracy =[]
predictions = []


for epoch in range(31):  # 30 epochs
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
        outputs = net(inputs).cuda()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 150 == 0:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss)) #print running loss divided by total number of mini-batches
    #create new i for if i - 79 append final running loss
    losses.append(running_loss/312)
    
    if epoch % 3 == 0:
        net.eval()
        correct = 0
        total = 0
        test_runningLoss = 0.0
        for i, data in enumerate(testDataLoader, 0):
            # get the inputs
            inputs = data['image']
            inputs = inputs.float()
            labels = data['label'].float()

            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs).cuda()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            test_runningLoss += loss.data[0]
            if i % 100 == 0:    # print every 1000 mini-batches
                print('test runnig loss %.3f' %(test_runningLoss)) #print running loss divided by total number of mini-batches
        #create new i for if i - 79 append final running loss

            predicted = outputs > 0.5
            predicted = predicted.data.cpu().numpy()
            if epoch == 30:
                predictions.append(predicted)
            else:
                pass
            total += labels.size(0)
            correct += (predicted.reshape(16) == labels.data).sum()
            
        print('Accuracy of the network on the 4000 test images: %d %%' % (
                100 * correct / total))

        testAccuracy.append((100 * correct / total))            
        testLosses.append(test_runningLoss/250)

print('Finished Training')


#Save out results
np.savez("testAccuraciesFinal", testAccuracy)
np.savez("allLossesFinal", losses)
np.savez("allTestLossesFinal", testLosses)


#Plot test accuracy
plt.plot(testAccuracy)
plt.title("Test Accuracy LeNet5: SGD, ReLU, 15 x 15")
plt.xlabel("Epoch")
plt.xlim(0,10)
plt.ylabel("Classification Accuracy")
plt.show()


# learning curve plotting function
def plotLearningCurve(trainSizes,trainMean, testMean,title):
    plt.plot(trainSizes, trainMean, color = 'blue', marker = 'o', markersize=5, label = 'Train BCE')

    plt.plot(trainSizes, testMean, color = 'green', marker= 's', linestyle = '--', markersize= 5, 
            label = 'Test BCE')
    plt.grid()
    plt.xlabel('Number of Training Epochs')
    plt.title(title)
    plt.ylabel('Running Binary Cross Entropy Loss')
    plt.legend(loc= 'best')

#plot learning curve
plt.figure(figsize= (5,5))
plotLearningCurve(np.arange(11), losses[::3], testLosses,'Learning Curve LeNet5: SGD, ReLU, 15 x 15')
plt.show()


#flatten predictions to print out classification report and confusion matrix
flatPredictions = np.asarray(predictions).flatten()
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, flatPredictions))
print(confusion_matrix(y_test, flatPredictions))

#function to plot confusion matrix
def plotConfusion(confMatObject):
    fig, ax = plt.subplots(figsize = (5,5))
    ax.matshow(confMatObject, cmap=plt.cm.Blues, alpha  = 0.3)
    for i in range(confMatObject.shape[0]):
        for j in range(confMatObject.shape[1]):
            ax.text(x = j, y=i, s= confMatObject[i,j], va = 'center', ha = 'center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()
confmat = confusion_matrix(y_test, flatPredictions)
plotConfusion(confmat)



