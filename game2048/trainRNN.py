import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
#from Model import Net
import time
import pandas as pd
import numpy as np
import csv
from RNN import RNN1
import tqdm

batch_size = 128
#loading data
data = pd.read_csv('./DATA.csv')
data = data.values
X= data[:,0:16]
#X = np.reshape(X, (-1,4.4))
Y = data[:,16]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1,shuffle=False)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)


traindataset = torch.utils.data.TensorDataset(X_train,Y_train)
testdataset = torch.utils.data.TensorDataset(X_test,Y_test)

trainloader = torch.utils.data.DataLoader(dataset=traindataset,
                                           batch_size=batch_size,
                                           shuffle=True,
)

testloader = torch.utils.data.DataLoader(dataset=testdataset,
                                          batch_size=batch_size,
                                          shuffle=False
)


sequence_length = 16  # 序列长度，将图像的每一列作为一个序列
input_size = 1  # 输入数据的维度
hidden_size = 256  # 隐藏层的size
num_layers = 4  # 有多少层

num_classes = 4
batch_size = 128
NUM_EPOCHS = 8
learning_rate = 0.0001

model = RNN1(input_size, hidden_size, num_layers, num_classes)
model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)


for epoch in range(NUM_EPOCHS):
    i = 0
    running_loss = 0
    print('EPOCHS', epoch + 1)
    correct = 0
    for i, (images,labels) in enumerate(trainloader):
        images, labels = Variable(images), Variable(labels)
        labels = labels.long()
        optimizer.zero_grad()
        images = images.reshape(-1, 16,1).cuda()
        output = model(images).reshape(-1, 4).cuda()
        labels = labels.float().reshape(-1).cuda()
        correct += (labels.cpu().numpy() == output.cpu().detach().numpy().argmax(axis = 1)).sum()
        loss = criterion(output, labels.long())
        if i % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data), len(trainloader.dataset),
                       100. * i / len(trainloader), loss.item()))
        running_loss += float(loss)
        loss.backward()
        optimizer.step()
    print(running_loss/i)
    print("train accuracy: ", correct/float(X_train.shape[0]))
    '''
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = Variable(images), Variable(labels)
            labels = labels.long()
            images = images.reshape(-1, 16, 1).cuda()
            output = model(images).reshape(-1, 4).cuda()
            labels = labels.float().reshape(-1).cuda()
            correct += (labels.cpu().numpy() == output.cpu().detach().numpy().argmax(axis=1)).sum()
    print("test accuracy: ", correct / float(X_test.shape[0]))
   '''
torch.save(model.state_dict(),'modelrnn.pth' )