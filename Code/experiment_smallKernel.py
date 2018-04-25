'''This script is for training and test a LeNet5 with smaller kernel sizes: 3x3
5x5, 7x7 and then plotting out the test accuracy and training loss for each of
these network architectures'''

'''LeNet in PyTorch.''' 
## This class allows for calling of specific kernel sizes if first layer height/width
## is also specified
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LeNet(nn.Module):
    def __init__(self,kernel_size, firstLayer_size):
        super(LeNet, self).__init__()
        self.kernel = kernel_size
        self.firstLayer = firstLayer_size

        self.conv1 = nn.Conv2d(3, 6, int(self.kernel))
        self.conv2 = nn.Conv2d(6, 16, int(self.kernel))
        #starting at 224 with max pooling kernel of 2 and stride of 2 results in 53x53 
        #and 16 channels by the time we arrive at fully connected layer
        
        self.firstLayerHeightWidth = 16 * (self.firstLayer**2)
        
        self.fc1   = nn.Linear(int(self.firstLayerHeightWidth), 120)
        print(self.fc1)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 1)
        
    def forward(self, x):
#         print(x.shape)
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, kernel_size=2, stride=2)
#         print(out.size)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        #Note: i use sigmoid here and one output, but I am going to update this 
        #for softmax, which is better because it doesn't use a threshold. To do that
        #I need to change the y labels to sets of vectors that are [0,1] or [1,0]
        out = F.sigmoid(self.fc3(out))
        return out


import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


kernel_sizes = [3,5,7] 
firstLayerSize = [54,53,51] 


####### This code block iterates through kernel size and firstlayer size specifications
# using the LeNet 5 and captures training loss and test accuracy after every epoch ####
allLosses = []
testAccuracies =[]
for kernel, layersize in zip(kernel_sizes, firstLayerSize):
    net = LeNet(kernel, layersize).cuda()
    
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    losses = []
    testAccuracy =[]

    
    for epoch in range(10):  # loop over the dataset multiple times
#         net.train()
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
            if i % 25 == 0:    # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss)) #print running loss divided by total number of mini-batches
        #create new i for if i - 79 append final running loss
        losses.append(running_loss/79)
        
        
        correct = 0
        total = 0
        for data in testDataLoader:
            images = data['image'].float()
            labels = data['label'].type(torch.FloatTensor)
            outputs = net(Variable(images).cuda())
#             _, predicted = torch.max(outputs.data, 1)
            predicted = outputs > 0.5
            predicted = predicted.data.cpu().numpy()
#             print(label)
#             print(predicted)
            total += labels.size(0)
            correct += (predicted.reshape(16) == labels.numpy()).sum()

        print('Accuracy of the network on the 4000 test images: %d %%' % (
            100 * correct / total))
        #say accuracy = a.ppend(100*correct/total and move testAccuracies.append of accuracy outside of to where allLosses is)
        testAccuracy.append((100 * correct / total))
    testAccuracies.append(testAccuracy)
    allLosses.append(losses)
# testAccuracies.append(testAccuracy)
print('Finished Training')


#save the testaccuracies and train losses so they can be accessed later 
np.savez("testAccuracies", testAccuracies)
np.savez("allLosses", allLosses)

###load if already saved
# import numpy as np
# testAccuracies = np.load('testAccuracies.npz')['arr_0']
# allLosses = np.load('allLosses.npz')['arr_0']


####This code block plots training loss and test accuracies by kernel size #####
import matplotlib.pyplot as plt
kernels = ["3x3", "5x5", "7x7"]
plt.figure(figsize=(15,7))
plt.subplot(1,2,1)
for i in range(len(allLosses)):
    s = kernels[i]
    plt.plot(allLosses[i], label = s)
plt.title("Training Loss by Convolution Kernel Size")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy Loss")
plt.xlim(0,10)
plt.legend(kernels, loc = 1)

plt.subplot(1,2,2)
for i in range(len(testAccuracies)):
    s = kernels[i]
    plt.plot(testAccuracies[i],label = s)
plt.title("Test Accuracy by Convolution Kernel Size")
plt.xlabel("Epoch")
plt.xlim(0,10)
plt.ylabel("Classification Accuracy")
plt.legend((kernels), loc = 'best')
plt.show()