# -*- coding: utf-8 -*-
# !/usr/bin/env python

# --------------------------------------------------------
# reinforce-deelp-learning-flappybird
# Copyright (c) 2016 SLXrobot
# Written by Chao Yu
# github : foolyc
# --------------------------------------------------------

import sys

FLG_GPU = True
GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0000 # final value of epsilon
INITIAL_EPSILON = 0.0 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
FRAME = 4
HEIGHT = 80
WIDTH = 80

UPDATE_STEP = 100
SAVE_STEP = 10000

sys.path.append("game/")
