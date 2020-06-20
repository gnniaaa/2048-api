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
from Net import CNNet


batch_size = 128
NUM_EPOCHS = 10

#loading data
data = pd.read_csv('./all2.csv')
data = data.values
X= data[:,0:16]
X = np.reshape(X, (-1,4,4))
Y = data[:,16]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,shuffle=False)
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

model=CNNet()
optimizer=optim.Adam(model.parameters(),lr=0.001)
criterion =torch.nn.CrossEntropyLoss()

def train():
    for epoch in range(NUM_EPOCHS):
        train_loss = 0
        for i, (data,label) in enumerate(trainloader):
            data = data.type(torch.float)
            if torch.cuda.is_available():
                data = Variable(data).cuda()
                target = Variable(target).cuda()
                moel.cuda()
            data = data.unsqueeze(dim=1)
            optimizer.zero_grad() #梯度清零
            output = model(data)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step() #参数更新
            if i % 1000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(data), len(trainloader.dataset),
                           100. * i / len(trainloader), loss.item()))

        model.eval()
        train_loss = 0
        correct_train = 0
        test_loss = 0
        correct_test = 0
        with torch.no_grad():
            for data, label in trainloader:
                data = data.unsqueeze(dim=1)
                output = model(data)  # 正向计算预测值
                pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
                correct_train += pred.eq(label.view_as(pred)).sum().item()  # 找到正确的预测值
            for data, label in testloader:
                data = data.unsqueeze(dim=1)
                output = model(data)  # 正向计算预测值
                pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
                correct_test += pred.eq(label.view_as(pred)).sum().item()
        torch.save(model.state_dict(),'model.pth' )
        train_accuracy = 100. * correct_train / len(trainloader.dataset)
        test_accuracy = 100. * correct_test / len(testloader.dataset)
        print('Training accuracy: %0.2f%%' % (train_accuracy))
        print('Testing accuracy: %0.2f%%' % (test_accuracy))

    print('Training accuracy: %0.2f%%' % (train_accuracy))
    print('Testing accuracy: %0.2f%%' % (test_accuracy))

if __name__=='__main__':
	train()
