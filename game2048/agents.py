import numpy as np
from game2048.RNN import RNN1
from game2048.Net import CNNet
import os
import torch
from torchvision import transforms
from sklearn import preprocessing

input_size = 11  # 输入数据的维度
hidden_size = 256  # 隐藏层的size
num_layers = 4  # 有多少层

num_classes = 4
batch_size = 128
num_epochs = 20
learning_rate = 0.001

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from game2048.expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

class MyAgent(Agent):

    def __init__(self, game, display=None):
        super().__init__(game, display)
        Path = os.getcwd() + '/game2048/model256.pth'
        model = CNNet()
        model.load_state_dict(torch.load(Path))
        self.search_func = model

    def step(self):
        board = self.game.board
        board[board == 0] = 1
        board = np.log2(board).flatten()
        board = board.reshape((4, 4))
        board = board[:, :,np.newaxis]
        board = transforms.ToTensor()(board)
        board = torch.unsqueeze(board, dim=0).float()
        output = self.search_func(board)
        direction = output.data.max(1, keepdim=True)[1]
        return direction

class MyAgent1(Agent):
    def __init__(self, game, display=None):
        Path = os.getcwd() + '/game2048/modelxin4.pth'
        self.game = game
        self.display = display
        self.model1 = RNN1(1, 256, 4, 4)
        self.model1.load_state_dict(torch.load(Path,map_location='cpu'))
        self.model1 = self.model1


    def step(self):
        board = self.game.board
        board[board == 0] = 1
        board = np.log2(board).flatten()
        board = torch.FloatTensor(board)
        board = board.reshape((-1, 16,1))
        board = board
        output = self.model1(board).reshape(-1, 4)
        direction = output.cpu().detach().numpy().argmax(axis = 1)
        a=direction[0]
        print(a)

        return a

class MyAgent2(Agent):
    def __init__(self, game, display=None):
        Path1 = os.getcwd() + '/game2048/modelrnn.pth'
        Path2 = os.getcwd() + '/game2048/modelrnn1.pth'
        Path3 = os.getcwd() + '/game2048/modelrnn2.pth'
        Path4 = os.getcwd() + '/game2048/modelxin3.pth'
        Path5 = os.getcwd() + '/game2048/modelxin1.pth'
        Path6 = os.getcwd() + '/game2048/modelxin2.pth'
        Path7 = os.getcwd() + '/game2048/modelxin4.pth'
        self.game = game
        self.display = display
        self.model1 = RNN1(1, 256, 4, 4)
        self.model2 = RNN1(1, 256, 4, 4)
        self.model3 = RNN1(1, 256, 4, 4)
        self.model4 = RNN1(1, 256, 4, 4)
        self.model5 = RNN1(1, 256, 4, 4)
        self.model6 = RNN1(1, 256, 4, 4)
        self.model7 = RNN1(1, 256, 4, 4)

        self.model1.load_state_dict(torch.load(Path1,map_location='cpu'))
        self.model2.load_state_dict(torch.load(Path2, map_location='cpu'))
        self.model3.load_state_dict(torch.load(Path3, map_location='cpu'))
        self.model4.load_state_dict(torch.load(Path4, map_location='cpu'))
        self.model5.load_state_dict(torch.load(Path5, map_location='cpu'))
        self.model6.load_state_dict(torch.load(Path6, map_location='cpu'))
        self.model7.load_state_dict(torch.load(Path7, map_location='cpu'))
        self.model1 = self.model1


    def step(self):
        board = self.game.board
        board[board == 0] = 1
        board = np.log2(board).flatten()
        board = torch.FloatTensor(board)
        board = board.reshape((-1, 16,1))
        output1 = self.model1(board).reshape(-1, 4)
        output2 = self.model2(board).reshape(-1, 4)
        output3 = self.model3(board).reshape(-1, 4)
        output4 = self.model4(board).reshape(-1, 4)
        output5 = self.model5(board).reshape(-1, 4)
        output6 = self.model6(board).reshape(-1, 4)
        output7 = self.model7(board).reshape(-1, 4)
        direction1 = output1.cpu().detach().numpy().argmax(axis = 1)
        direction2 = output2.cpu().detach().numpy().argmax(axis=1)
        direction3 = output3.cpu().detach().numpy().argmax(axis=1)
        direction4 = output4.cpu().detach().numpy().argmax(axis=1)
        direction5 = output5.cpu().detach().numpy().argmax(axis=1)
        direction6 = output6.cpu().detach().numpy().argmax(axis=1)
        direction7 = output7.cpu().detach().numpy().argmax(axis=1)
        a=np.zeros(4)
        a[direction1[0]] += 1;
        a[direction2[0]] += 1;
        a[direction3[0]] += 1;

        a[direction4[0]] += 1;
        a[direction5[0]] += 1;
        a[direction6[0]] += 1;
        a[direction7[0]] += 1;



        return np.argmax(a)

class MyAgent3(Agent):
    def __init__(self, game, display=None):
        Path1 = os.getcwd() + '/game2048/modelxin4.pth'
        Path2 = os.getcwd() + '/game2048/model2566.pth'
        self.game = game
        self.display = display
        self.model1 = RNN1(1, 256, 4, 4)
        self.model1.load_state_dict(torch.load(Path1,map_location='cpu'))
        self.model1 = self.model1
        self.model2 = RNN1(1, 256, 4, 4)
        self.model2.load_state_dict(torch.load(Path2, map_location='cpu'))
        self.model2 = self.model2


    def step(self):
        board = self.game.board
        board[board == 0] = 1
        board = np.log2(board).flatten()
        board = torch.FloatTensor(board)
        board = board.reshape((-1, 16,1))
        board = board
        if np.amax(self.game.board)>=128:
          output = self.model1(board).reshape(-1, 4)
        else:
            output = self.model2(board).reshape(-1, 4)
        direction = output.cpu().detach().numpy().argmax(axis = 1)
        a=direction[0]
        print(a)

        return a