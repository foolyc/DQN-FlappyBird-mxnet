# A Simple Version of DQN Flappy Bird in mxnet (Nature version)

## Overview
This project follows the description of the Deep Q Learning algorithm described in [2]. Comparing to DQN algorithm in [1], **the target net(used to caculate the target Q value) was periodically update ** rather than updated each step

## Dependency:
* Python 2.7
* mxnet
* pygame
* OpenCV-Python

## Usage

- download the source code

```
git clone https://github.com/foolyc/DQN-FlappyBird-mxnet
```

- train the model 


```
python flappybird.py train -p './snapshot/pre.params'
```

- play the game with a pretrained model file

```
python flappybird.py test -p './snapshot/pre.params'
```

- configuration

```
FLG_GPU = True # using gpu or cpu
GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0000 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1 # number of frames to skip
FRAME = 4 # number of past frames to use as the input data of the q net
HEIGHT = 80 # height of input image
WIDTH = 80 # width of input image
UPDATE_STEP = 100 # target net updating period
SAVE_STEP = 10000 # saving the params per step period
```


## References

[1] Mnih Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. **Human-level Control through Deep Reinforcement Learning**. Nature, 529-33, 2015.

[2] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. **Playing Atari with Deep Reinforcement Learning**. NIPS, Deep Learning workshop


## Disclaimer
highly based on the following repos:

1. [yenchenlin/DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)
