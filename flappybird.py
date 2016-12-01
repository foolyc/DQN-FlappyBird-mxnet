# -*- coding: utf-8 -*-
# !/usr/bin/env python

# --------------------------------------------------------
# reinforce-deelp-learning-flappybird
# Copyright (c) 2016 SLXrobot
# Written by Chao Yu
# github : foolyc
# --------------------------------------------------------
from config import *
import cv2
import sys
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import argparse

import mxnet as mx


class DQNBird(object):
    def __init__(self):
        self.replayMemory = deque()
        self.timestep = 0
        if FLG_GPU:
            self.ctx = mx.gpu()
        else:
            self.ctx = mx.cpu()
        # if args.mode == "train":
        pre_model = None
        if 1:
            self.q_net = mx.mod.Module(symbol=self.createNet(1), data_names=['frame', 'act_mul'], label_names=['target', ], context=self.ctx)
            self.q_net.bind(data_shapes=[('frame', (BATCH, FRAME, HEIGHT, WIDTH)), ('act_mul', (BATCH, ACTIONS))], label_shapes=[('target', (BATCH,))], for_training=True)
            self.q_net.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34), arg_params=pre_model)
            self.q_net.init_optimizer(optimizer='adam', optimizer_params={'learning_rate': 0.0002, 'wd': 0.0, 'beta1': 0.5})
            shape = {"frame": (BATCH, FRAME, HEIGHT, WIDTH), 'act_mul':(BATCH, ACTIONS), 'target': (BATCH,)}
            dt = mx.viz.plot_network(symbol=self.q_net.symbol, shape=shape)
            dt.view()

        self.tg_net = mx.mod.Module(symbol=self.createNet(), data_names=['frame',], label_names=[], context=self.ctx)
        self.tg_net.bind(data_shapes=[('frame', (1, FRAME, HEIGHT, WIDTH))], for_training=False)
        self.tg_net.init_params(initializer=mx.init.Xavier(factor_type='in', magnitude=2.34), arg_params=pre_model)


    def createNet(self, type=0):
        '''
        :param type:
        type:0 predicted net
            1 trained net
        :return:
        '''
        frame = mx.symbol.Variable('frame')
        if type:
            act_mul = mx.sym.Variable('act_mul')
            target = mx.sym.Variable('target')

        conv1 = mx.sym.Convolution(data=frame, kernel=(8, 8), stride=(4, 4), pad=(2, 2), num_filter=32, name='conv1')
        relu1 = mx.sym.Activation(data=conv1, act_type='relu', name='relu1')
        pool1 = mx.sym.Pooling(data=relu1, kernel=(2, 2), stride=(2, 2), pool_type='max', name='pool1')

        conv2 = mx.sym.Convolution(data=pool1, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=64, name='conv2')
        relu2 = mx.sym.Activation(data=conv2, act_type='relu', name='relu2')
        conv3 = mx.sym.Convolution(data=relu2, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=64, name='conv3')
        relu3 = mx.sym.Activation(data=conv3, act_type='relu', name='relu3')

        flat = mx.sym.Flatten(data=relu3, NameError='flat')
        fc4 = mx.sym.FullyConnected(data=flat, num_hidden=512, name='fc4')
        relu4 = mx.sym.Activation(data=fc4, act_type='relu', name='relu4')
        fc5 = mx.sym.FullyConnected(data=relu4, num_hidden=ACTIONS, name='fc5')
        if not type:
            return fc5
        else:
            q_act = mx.sym.sum(fc5 * act_mul, axis=1, name='q_act')
            output = (q_act - target) ** 2

            loss = mx.sym.MakeLoss(output)
            return loss


    def trainStep(self):
        minibatch = random.sample(self.replayMemory, BATCH)
        state_batch = np.squeeze([data[0] for data in minibatch])
        filter_batch = np.squeeze([data[1] for data in minibatch])
        target_batch = np.squeeze([data[2] for data in minibatch])
        nextState_batch = [data[3] for data in minibatch]
        terminal_batch = np.squeeze([data[4] for data in minibatch])

        qvalue = []
        for i in range(BATCH):
            input_frame = np.reshape(nextState_batch[i],(1, FRAME, HEIGHT, WIDTH))
            self.tg_net.forward(mx.io.DataBatch([mx.nd.array(input_frame, self.ctx)], []))
            qvalue.append(self.tg_net.get_outputs()[0].asnumpy())
        qvalue_batch = np.squeeze(qvalue)


        target_batch[terminal_batch == 0] += GAMMA * np.max(qvalue_batch, axis=1)[terminal_batch == 0]
        self.q_net.forward(mx.io.DataBatch([mx.nd.array(state_batch, self.ctx), mx.nd.array(filter_batch, self.ctx)], [mx.nd.array(target_batch, self.ctx)] ), is_train=True)
        self.q_net.backward()
        self.q_net.update()

        if self.timestep % SAVE_STEP == 0:
            self.q_net.save_params('./snapshot/iter_%5d.params'%(self.timestep))
        if self.timestep % UPDATE_STEP ==0:
            arg_params, aux_params = self.q_net.get_params()
            print "update target network......."
            self.tg_net.init_params(initializer=None, arg_params=arg_params, aux_params=aux_params, force_init=True)



    def trainNet(self):
        game_state = game.GameState()
        # get the first state by doing nothing and preprocess the image to 80x80x4
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        x_t, r_0, terminal = game_state.frame_step(do_nothing)
        x_t = cv2.cvtColor(cv2.resize(x_t, (HEIGHT, WIDTH)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
        s_t = np.zeros([FRAME, HEIGHT, WIDTH])
        for i in range(FRAME):
            s_t[i, :, :] = x_t
        # saving and loading networks
        # start training
        epsilon = INITIAL_EPSILON
        t = 0
        while "flappy bird" != "angry bird":
            input_frame = np.reshape(s_t, (1, FRAME, HEIGHT, WIDTH))
            self.tg_net.forward(mx.io.DataBatch([mx.nd.array(input_frame, self.ctx)], []))
            qvalue = np.squeeze(self.tg_net.get_outputs()[0].asnumpy())
            a_t = np.zeros([ACTIONS])
            action_index = 0
            if t % FRAME_PER_ACTION == 0:
                if random.random() <= epsilon:
                    # print("----------Random Action----------")
                    action_index = random.randrange(ACTIONS)
                    a_t[action_index] = 1
                else:
                    action_index = np.argmax(qvalue)
                    a_t[action_index] = 1
                    # print "----------Net Action----------", action_index
            else:
                a_t[0] = 1 # do nothing


            # scale down epsilon
            if epsilon > FINAL_EPSILON and self.timestep > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            # run the selected action and observe next state and reward
            x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (HEIGHT, WIDTH)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (1, HEIGHT, WIDTH))
            s_t1 = np.vstack((x_t1, s_t[:(FRAME-1), :, :]))

            # store the transition in D
            self.replayMemory.append((s_t, a_t, r_t, s_t1, terminal))



            if len(self.replayMemory) > REPLAY_MEMORY:
                self.replayMemory.popleft()
            # only train if done observing
            if self.timestep > OBSERVE:
                self.trainStep()
            s_t = s_t1
            self.timestep += 1
            if self.timestep <= OBSERVE:
                state = "observe"
            elif self.timestep > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"
            print "TIMESTEP", self.timestep, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q ", qvalue


    def playGame(self):
        pass


if __name__ == "__main__":
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('mode', default="train", help="train or test")
    # args = parser.parse_args()
    fb = DQNBird()
    fb.trainNet()
    # if args.mode == "train":
    #     # fb.trainNet()
    #     fb.trainNet()
    # else:
    #     fb.playGame()
