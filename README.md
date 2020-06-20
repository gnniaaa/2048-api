# 2048-api
A 2048 game api for training supervised learning (imitation learning) or reinforcement learning agents

# Code structure
* [`game2048/`](game2048/): the main package.
    * [`game.py`](game2048/game.py): the core 2048 `Game` class.
    * [`agents.py`](game2048/agents.py): the `Agent` class with instances.
    * [`displays.py`](game2048/displays.py): the `Display` class with instances, to show the `Game` state.
    * [`expectimax/`](game2048/expectimax): a powerful ExpectiMax agent by [here](https://github.com/nneonneo/2048-ai).
    * [`RNN.py`](game2048/RNN.py):RNN net
    * [`Net.py`](game2048/Net.py):CNN net
    * [`trainRNN.py`](game2048/trainRNN.py):offline train
    * [`zaixian2048.py`](game2048/zaixian2048.py):online train
    * [`zaixian256.py`](game2048/zaixian256.py):online train
* [`explore.ipynb`](explore.ipynb): introduce how to use the `Agent`, `Display` and `Game`.
* [`static/`](static/): frontend assets (based on Vue.js) for web app.
* [`webapp.py`](webapp.py): run the web app (backend) demo.
* [`evaluate.py`](evaluate.py): evaluate your self-defined agent.


# Requirements
* code only tested on linux system (ubuntu 16.04)
* Python 3 (Anaconda 3.6.3 specifically) with numpy and flask

# To define your own agents
```python
from game2048.agents import Agent

class YourOwnAgent(Agent):

    def step(self):
        '''To define the agent's 1-step behavior given the `game`.
        You can find more instance in [`agents.py`](game2048/agents.py).
        
        :return direction: 0: left, 1: down, 2: right, 3: up
        '''
        direction = some_function(self.game)
        return direction

```
# run
* run trainRNN.py or zaixian2048.py or zaixian256.py to obtain model
* run evaluate.py to get score
* testAgent can be changed
    * MyAgent1:single model
    * MyAgent2:multiple models to vote
    * MyAgent3:2 models individually used under 258 and above 256

# To compile the pre-defined ExpectiMax agent

```bash
cd game2048/expectimax
bash configure
make
```

# To run the web app
```bash
python webapp.py
```


# LICENSE
The code is under Apache-2.0 License.

# For EE369 students from SJTU only
Please read course project [requirements](EE369.md) and [description](Project2048.pdf). 
