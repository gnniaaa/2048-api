import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1,2)) #32, 4x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(2, 1)) #64, 3x3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3),padding=(2,2)) #128, 5x5
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(1, 1)) #128, 5x5
        self.conv5 = nn.Conv2d(128, 64, kernel_size=(2, 2)) #64,4x4
        self.fc1 = nn.Linear(64 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 4)
        self.batch_norm1 = nn.BatchNorm1d(64 * 4 * 4)
        self.batch_norm2 = nn.BatchNorm1d(1024)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.batch_norm4 = nn.BatchNorm1d(64)
        self.drop = nn.Dropout2d(p=0.5)
        self.initialize()

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.drop(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 64 * 4 * 4)
        x = self.drop(x)
        #x = self.batch_norm1(x)
        x = F.relu(self.fc1(x))
        #x = self.batch_norm2(x)
        x = F.relu(self.fc2(x))
        #x = self.batch_norm3(x)
        x = F.relu(self.fc3(x))
        #x = self.batch_norm4(x)
        x = self.fc4(x)

        return F.log_softmax(x)

        #return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')

class CNNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(4, 1), padding=(2, 0)) #64, 5x4
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 4), padding=(0, 2)) #128, 5x5
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(2, 2)) #128, 4x4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)) #128, 4x4
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(4, 4), padding=(2, 2)) #128, 5x5
        self.conv6 = nn.Conv2d(128,128,kernel_size=(2,2)) #128, 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 4)
        self.batch_norm1 = nn.BatchNorm1d(128 * 4 * 4)
        self.batch_norm2 = nn.BatchNorm1d(2048)
        self.batch_norm3 = nn.BatchNorm1d(512)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.view(-1, 128 * 4 * 4)
        x = self.batch_norm1(x)
        x = F.relu(self.fc1(x))
        x = self.batch_norm2(x)
        x = F.relu(self.fc2(x))
        x = self.batch_norm3(x)
        x = self.fc3(x)

        return x


