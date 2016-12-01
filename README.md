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


## References

[1] Mnih Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. **Human-level Control through Deep Reinforcement Learning**. Nature, 529-33, 2015.

[2] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. **Playing Atari with Deep Reinforcement Learning**. NIPS, Deep Learning workshop


## Disclaimer
highly based on the following repos:

1. [yenchenlin/DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)
2. [li-haoran/DRL-FlappyBird](https://github.com/li-haoran/DRL-FlappyBird)