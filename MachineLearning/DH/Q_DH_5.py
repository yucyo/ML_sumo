import math
import sys
import os
import random

import numpy as np
import pygame
from pygame.locals import*

import tensorflow as tf

import DH_5

env = DH_5.DemonHuman(step=True,image=False)
np_actions = 3
