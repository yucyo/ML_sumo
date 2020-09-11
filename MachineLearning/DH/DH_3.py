import numpy as np
import math
import tensorflow as tf
import pygame
from pygame.locals import *
from collections import deque
import random
import sys
import gym

pygame.init()
screen = pygame.display.set_mode((1000,1000))
pygame.display.set_caption("TestSeeking")
font=pygame.font.SysFont('Calibri',25,True,False)
clock = pygame.time.Clock()


dx=400
dy=400
hx=0
hy=0
score=0
timeRemain=600
Oni=[int(dx),int(dy),30,30]
Hito=[100,100,30,30]

done = False
while not done:

    screen.fill((255,255,255))

    for event in pygame.event.get():
        if event.type == pygame.quit:
            done = True
        # if event.key == K_ESCAPE:
        #     done = True
        if event.type == KEYDOWN:
            if event.key == K_LEFT:
                dx -= 15
            if event.key == K_RIGHT:
                dx += 15
            if event.key == K_UP:
                dy -= 15
            if event.key == K_DOWN:
                dy += 15
            Oni=[dx,dy,30,30]
            # if event.type == K_:
        #     u , v = j.get_axis(0), j.get_axis(1)
        #     if int(u*10)+int(v*10) != 0:
        #         x= x + int(u*10)
        #         y= y + int(v*10)
        #         if x > 100:
        #             x= 100
        #         if x < -100:
        #             x= -100
        #         if y > 80:
        #             y= 80
        #         if y < -80:
        #             y= -80
        #         posRect= [120-10+x, 90-10+y, 20, 20]


    r= np.random.rand(2)
    speed= 1.0
    hx= hx + int(r[0]*20-10)*speed
    hy= hy + int(r[1]*20-10)*speed
    if hx > 1000:
        hx= 1000
    if hx < -1000:
        hx= -1000
    if hy > 1000:
        hy= 1000
    if hy < -1000:
        hy= -1000
    Hito= [100+hx, 100+hy, 30, 30]

    if (hx - dx)**2 + (hy - dy)**2 < 1000:
        score = score + 1

    pygame.draw.rect(screen, (255,0,0), Hito, 2)
    pygame.draw.rect(screen, (0,0,255), Oni, 2)

    timeRemain= timeRemain - 1
    text= font.render(str(score)+' '+str(timeRemain),True,(0,0,0))
    screen.blit(text,[0,0])
    pygame.display.flip()

    if timeRemain == 0:
        pygame.time.wait(10000)
        done = True
    elif score == 10:
        pygame.time.wait(10000)
        done = True

    clock.tick(10)

pygame.quit()
