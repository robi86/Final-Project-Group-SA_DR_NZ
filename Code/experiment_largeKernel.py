'''This script is for training and testing a LeNet5 with large kernel sizes:10x10
15x15, 20x20 and then plotting out the training and test losses for each of
these network architectures at each epoch as well as the test accuracies'''

#specify kernel sizes and layer sizes
kernel_sizes = [10, 15,20]
firstLayerSize = [49, 45,41] 

#create lists to capture losses and accuracies
allTrainLosses = []
allTestLoses = []
testAccuracies =[]

#iterate through kernel sizes
for kernel, layersize in zip(kernel_sizes, firstLayerSize):
    net = LeNet(kernel, layersize).cuda()
    
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    losses = []
    testLosses = []
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
            outputs = net(inputs).cuda()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 100 == 0:    # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss)) #print running loss divided by total number of mini-batches
        #create new i for if i - 79 append final running loss
        losses.append(running_loss/312)
        
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
            if i % 50 == 0:    # print every 1000 mini-batches
                print('test runnig loss %.3f' %(test_runningLoss)) #print running loss divided by total number of mini-batches
        #create new i for if i - 79 append final running loss
        
            predicted = outputs > 0.5
            predicted = predicted.data.cpu().numpy()
            total += labels.size(0)
            correct += (predicted.reshape(16) == labels.data).sum()

        print('Accuracy of the network on the 4000 test images: %d %%' % (
            100 * correct / total))

        testAccuracy.append((100 * correct / total))            
        testLosses.append(test_runningLoss/250)
    testAccuracies.append(testAccuracy)
    allTestLoses.append(testLosses)
    allTrainLosses.append(losses)
        # testAccuracies.append(testAccuracy)
print('Finished Training')


#save the testaccuracies and train losses so they can be accessed later 
np.savez("testAccuraciesLarge", testAccuracies)
np.savez("allLossesLarge", allTrainLosses)
np.savez("allTestLossesLarge", allTestLoses)


# plot allTestLoses, allTrainLosses, testAccuracies
import matplotlib.pyplot as plt
kernels = ["10x10", "15x15", "20x20"]
plt.figure(figsize=(15,7))
plt.subplot(1,2,1)
for i in range(len(allTrainLosses)):
    s = kernels[i]
    plt.plot(allTrainLosses[i], label = s)
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


#### This block of code plots a learning curve which shows the difference 
#### between training and test loss at for each epoch
def plotLearningCurve(trainSizes,trainMean, testMean,title):
    plt.plot(trainSizes, trainMean, color = 'blue', marker = 'o', markersize=5, label = 'training_accuracy')

    plt.plot(trainSizes, testMean, color = 'green', marker= 's', linestyle = '--', markersize= 5, 
            label = 'validation_accuracy')
    plt.grid()
    plt.xlabel('Number of Training Epochs')
    plt.title(title)
    plt.ylabel('Running Binary Cross Entropy Loss')
    plt.legend(loc= 'best')

#     plt.show()
plt.figure(figsize= (20,7))
for i in range(3):
    plt.subplot(1,3,i+1)
    plotLearningCurve(np.arange(10), allTrainLosses[i], allTestLoses[i],'LeNet5 CNN w/ %s Kernel Size' % kernels[i])
plt.show()
