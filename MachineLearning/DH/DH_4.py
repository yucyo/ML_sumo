#coding UTF-8
#DemonHumanDeepLearning

import numpy as np
from PIL umport Image
import matplotlib.pyplot as plt
import tensorflow as tf

import pygame
from pygame.locals import *

import os
import sys
import math
import random
#[0-1]放射状感知関数(無理かも)
#[0-2]追従関数(いらないかも)
#[0-3]逃亡関数(いらないかも)
#[0-4]H報酬関数
def rewardH():

    if (hx - dx)**2 + (hy - dy)**2 < 1000:
        reward = -10000
    else if hx0 == hx and hy0 ==hy:
        reward = -10
    else :
        reward = +20

#[0-5]D報酬関数
def rewardD():

    if (hx - dx)**2 + (hy - dy)**2 < 1000:
        reward = +10000
    else if dx0 == dx and dy0 ==dy:
        reward = -10
#[0-6]DH初期位置
Demon=[dx0,dy0,10,10]
Human=[hx0,hy0,10,10]

#[0-7]モデル設計
model = models.Sequential()
model.add()





#[1]Q関数を設計する
#[2]行動a(s)を求める関数
#[3]Qテーブルを更新する関数
#[4]メイン関数開始・パラメータ設定
env = gym.make("DH_4")
q_table = np.random.uniform()
#[5]メインルーチン
def step(self, action):
    if not self.step_flag:
        sys.stdout.write("Not running in stepping mode!\n")
        return None

    reward = 0
    done = False
    self._key_action(action)
    self.update()
    self.draw(self.screen)
    pygame.display.update()
    self._get_observation()
    self.no_kill_counter += 1

    if (hx - dx)**2 + (hy - dy)**2 < 1000:
        reward = +10000
    elif dx0 == dx and dy0 ==dy:
        reward = -10

    elif self.counter == 4500:
        reward += -1000
        done = True
        sys.stdout.write("Time up!\n")
    return self.observation_space, reward, done, dict()
