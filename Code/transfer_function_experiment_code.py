


'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #starting at 224 with max pooling kernel of 2 and stride of 2 results in 53x53 
        #and 16 channels by the time we arrive at fully connected layer
        
        self.fc1   = nn.Linear(16*53*53, 120)  
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
        #Note: i use sigmoid here and one output, but I am going to update this 
        #for softmax, which is better because it doesn't use a threshold. To do that
        #I need to change the y labels to sets of vectors that are [0,1] or [1,0]
        out = F.sigmoid(self.fc3(out))
        return out
net = LeNet().cuda()


# In[26]:

import torch.optim as optim

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# ### Optimize

# In[33]:

import time


# In[34]:

start = time.time()

t_running_loss=[]
t_accuracy=[]

for epoch in range(10):  # loop over the dataset multiple times
    #model.train()
    #call ceiling on number of mini batches 
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
        if i % 1000 == 0:    # print every 1000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss/1000)) #print running loss divided by total number of mini-batches
#             running_loss = 0.0 i dont just want my running loss at the end of each epoch rather than every minibatch
        #model.eval() - also negates val is for tuning of hyperparameter and know when to stop training
    print(running_loss/1000)
    r_l_k=(running_loss/1000)
    t_running_loss.insert(0, r_l_k)
    t_running_loss_2 = t_running_loss[::-1]
    
    #to get accuracy by epoch, put it inside this loop


# In[37]:

from sklearn.metrics import f1_score

correct = 0
total = 0
f1_scores = []
for data in testDataLoader:
    images = data['image'].float()
    labels = data['label'].type(torch.FloatTensor)
    outputs = net(Variable(images).cuda())
    predicted = outputs > 0.5
    predicted = predicted.data.cpu().numpy()
    
    total += labels.size(0)
    correct += (predicted.reshape(16) == labels.numpy()).sum()
    f1_scores.append(f1_score(labels.numpy(), predicted.reshape(16)))

total_f1 = pd.Series(f1_scores)
total_f1 = total_f1.mean()
print('Accuracy of the network on the 4000 test images: %d %%' % (
    100 * correct / total))
print('F1 Score: {}'.format(total_f1))


# In[38]:

accuracy=(100 * correct / total)
t_accuracy.insert(0, accuracy)
t_accuracy_2 = t_accuracy[::-1]


# In[40]:

ep=np.arange(0, 10, 1)

print(ep)
print(t_running_loss_2)
print(t_accuracy_2)

print ('Time Elapsed:', time.time() - start, 'seconds.')


# In[42]:

get_ipython().magic('matplotlib inline')


# In[47]:

plt.plot(ep, t_running_loss_2)
plt.ylabel('Training Loss Values')
plt.xlabel('Number of Epochs')

plt.show()


# #### Try with different transfer function:

# In[56]:

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #starting at 224 with max pooling kernel of 2 and stride of 2 results in 53x53 
        #and 16 channels by the time we arrive at fully connected layer
        
        self.fc1   = nn.Linear(16*53*53, 120)  
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 1)

    def forward(self, x):
        out = F.tanh(self.conv1(x))
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.tanh(self.conv2(out))
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = out.view(out.size(0), -1)
        out = F.tanh(self.fc1(out))
        out = F.tanh(self.fc2(out))
        #Note: i use sigmoid here and one output, but I am going to update this 
        #for softmax, which is better because it doesn't use a threshold. To do that
        #I need to change the y labels to sets of vectors that are [0,1] or [1,0]
        out = F.sigmoid(self.fc3(out))
        return out

tanh_net = LeNet().cuda()


# In[58]:

start = time.time()

t_running_loss=[]
t_accuracy=[]

for epoch in range(10):  # loop over the dataset multiple times
    #model.train()
    #call ceiling on number of mini batches 
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
        outputs = tanh_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 1000 == 0:    # print every 1000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss/1000)) #print running loss divided by total number of mini-batches
#             running_loss = 0.0 i dont just want my running loss at the end of each epoch rather than every minibatch
        #model.eval() - also negates val is for tuning of hyperparameter and know when to stop training
    print(running_loss/1000)
    r_l_k=(running_loss/1000)
    t_running_loss.insert(0, r_l_k)
    t_running_loss_2 = t_running_loss[::-1]
    
    #to get accuracy by epoch, put it inside this loop


# In[59]:

correct = 0
total = 0
f1_scores = []
for data in testDataLoader:
    images = data['image'].float()
    labels = data['label'].type(torch.FloatTensor)
    outputs = tanh_net(Variable(images).cuda())
    predicted = outputs > 0.5
    predicted = predicted.data.cpu().numpy()
    
    total += labels.size(0)
    correct += (predicted.reshape(16) == labels.numpy()).sum()
    f1_scores.append(f1_score(labels.numpy(), predicted.reshape(16)))

total_f1 = pd.Series(f1_scores)
total_f1 = total_f1.mean()
print('Accuracy of the network on the 4000 test images: %d %%' % (
    100 * correct / total))
print('F1 Score: {}'.format(total_f1))


# In[60]:

accuracy=(100 * correct / total)
t_accuracy.insert(0, accuracy)
t_accuracy_2 = t_accuracy[::-1]


# In[61]:

ep=np.arange(0, 10, 1)

print(ep)
print(t_running_loss_2)
print(t_accuracy_2)

print ('Time Elapsed:', time.time() - start, 'seconds.')


# In[70]:

import matplotlib.ticker as mtick

fig, ax = plt.subplots()

plt.plot(ep, t_running_loss_2)
plt.ylabel('Training Loss Values')
plt.xlabel('Number of Epochs')

ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

plt.show()


# In[71]:

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #starting at 224 with max pooling kernel of 2 and stride of 2 results in 53x53 
        #and 16 channels by the time we arrive at fully connected layer
        
        self.fc1   = nn.Linear(16*53*53, 120)  
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 1)

    def forward(self, x):
        out = F.sigmoid(self.conv1(x))
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = F.sigmoid(self.conv2(out))
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = out.view(out.size(0), -1)
        out = F.sigmoid(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        #Note: i use sigmoid here and one output, but I am going to update this 
        #for softmax, which is better because it doesn't use a threshold. To do that
        #I need to change the y labels to sets of vectors that are [0,1] or [1,0]
        out = F.sigmoid(self.fc3(out))
        return out

sig_net = LeNet().cuda()


# In[72]:

start = time.time()

t_running_loss=[]
t_accuracy=[]

for epoch in range(10):  # loop over the dataset multiple times
    #model.train()
    #call ceiling on number of mini batches 
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
        outputs = sig_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 1000 == 0:    # print every 1000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss/1000)) #print running loss divided by total number of mini-batches
#             running_loss = 0.0 i dont just want my running loss at the end of each epoch rather than every minibatch
        #model.eval() - also negates val is for tuning of hyperparameter and know when to stop training
    print(running_loss/1000)
    r_l_k=(running_loss/1000)
    t_running_loss.insert(0, r_l_k)
    t_running_loss_2 = t_running_loss[::-1]

print('Finished Training')


# In[73]:

correct = 0
total = 0
f1_scores = []
for data in testDataLoader:
    images = data['image'].float()
    labels = data['label'].type(torch.FloatTensor)
    outputs = sig_net(Variable(images).cuda())
    predicted = outputs > 0.5
    predicted = predicted.data.cpu().numpy()
    
    total += labels.size(0)
    correct += (predicted.reshape(16) == labels.numpy()).sum()
    f1_scores.append(f1_score(labels.numpy(), predicted.reshape(16)))

total_f1 = pd.Series(f1_scores)
total_f1 = total_f1.mean()
print('Accuracy of the network on the 4000 test images: %d %%' % (
    100 * correct / total))
print('F1 Score: {}'.format(total_f1))


# In[74]:

accuracy=(100 * correct / total)
t_accuracy.insert(0, accuracy)
t_accuracy_2 = t_accuracy[::-1]


# In[75]:

ep=np.arange(0, 10, 1)

print(ep)
print(t_running_loss_2)
print(t_accuracy_2)

print ('Time Elapsed:', time.time() - start, 'seconds.')


# In[76]:

_, ax = plt.subplots()

plt.plot(ep, t_running_loss_2)
plt.ylabel('Training Loss Values')
plt.xlabel('Number of Epochs')

ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

plt.show()


# In[ ]:



