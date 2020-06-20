import os
from torch.autograd import Variable
from RNN import RNN1
import torch
import torch.nn as nn
batch_size = 128
NUM_CLASSES = 4  # 分类数目
NUM_EPOCHS = 20  # 训练的迭代次数
import pandas as pd
from game import Game
from displays import Display
from agents import ExpectiMaxAgent, MyAgent1
import numpy as np
from sklearn.model_selection import train_test_split

image = []
label = []

display1 = Display()
display2 = Display()

stop_number = 2048

data = pd.read_csv('./DATA.csv')
data = data.values
X= data[:,0:16]
Y = data[:,16]
image=np.reshape(X,(-1,4,4))
image=image.tolist()
label=Y.tolist()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01,shuffle=False)
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
hidden_size = 256  # 隐藏层
num_layers = 4  # 有多少层

num_classes = 4
batch_size = 128
NUM_EPOCHS = 6
learning_rate = 0.0001

model = RNN1(input_size, hidden_size, num_layers, num_classes)
model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
correct=0

for epoch in range(NUM_EPOCHS):
    i = 0
    running_loss = 0
    print('EPOCHS', epoch + 1)
    correct = 0
    for i, (images, labels) in enumerate(trainloader):

        images, labels = Variable(images), Variable(labels)
        labels = labels.long()
        optimizer.zero_grad()
        images = images.reshape(-1, 16, 1).cuda()
        output = model(images).reshape(-1, 4).cuda()
        labels = labels.float().reshape(-1).cuda()
        loss = criterion(output, labels.long())
        loss.backward()
        optimizer.step()
        correct += (labels.cpu().numpy() == output.cpu().detach().numpy().argmax(axis=1)).sum()
        if i % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(data), len(trainloader.dataset),
                           100. * i / len(trainloader), loss.item()))
    print("train accuracy: ", correct/float(X_train.shape[0]))
torch.save(model.state_dict(),'modelxin1.pth' )


count = 0


while count < 200:

    image = []
    label = []
    count = 0

    for k in range(0, 200):

        game = Game(4, score_to_win=2048, random=False)
        agent = ExpectiMaxAgent(game, display=display1)
        my_agent = MyAgent1(game, display=display2)

        while game.end == False:
            direction1 = agent.step()

            board = game.board
            temp = np.amax(board)
            board[board == 0] = 1
            board = np.log2(board).flatten()
            board = torch.FloatTensor(board)
            board = board.reshape((-1, 16, 1))
            board = board.cuda()
            output = model(board).reshape(-1, 4)
            direction2 = output.cpu().detach().numpy().argmax(axis=1)

            image.append(board.tolist())
            label.append(direction1)
            game.move(direction2[0])

        display1.display(game)


        if temp == 1024:
            count += 1
        print(count)

    if count > 150:
        break
    else:
        image = np.array(image)
        label = np.array(label)

        x_train, x_test, y_train, y_test = train_test_split(image, label, test_size=0.1, random_state=30)

        x_train = torch.FloatTensor(x_train)
        x_test = torch.FloatTensor(x_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)


        traindataset = torch.utils.data.TensorDataset(x_train, y_train)
        testdataset = torch.utils.data.TensorDataset(x_test, y_test)

        trainloader = torch.utils.data.DataLoader(dataset=traindataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  )
        testloader = torch.utils.data.DataLoader(dataset=testdataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  )


        correct = 0
        running_loss = 0


        for i, (images, labels) in enumerate(trainloader):
            epoch=count
            images, labels = Variable(images), Variable(labels)
            labels = labels.long()
            optimizer.zero_grad()
            images = images.reshape(-1, 16, 1).cuda()
            output = model(images).reshape(-1, 4).cuda()
            labels = labels.float().reshape(-1).cuda()
            correct += (labels.cpu().numpy() == output.cpu().detach().numpy().argmax(axis=1)).sum()
            loss = criterion(output, labels.long())
            if i % 1000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(data), len(trainloader.dataset),
                           100. * i / len(trainloader), loss.item()))
            running_loss += float(loss)
            loss.backward()
            optimizer.step()
        print("train accuracy: ", correct / float(X_train.shape[0]))
        torch.save(model.state_dict(), 'modelxin1.pth')
