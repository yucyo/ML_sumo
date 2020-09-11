import numpy as np
import pygame
from pygame.locals import *
import os
from collections import deque
import temsorflow as tf

pygame.init()
screen = pygame.display.set_mode((1000,1000))
pygame.display.set_caption("TestSeeking")
font=pygame.font.SysFont('Calibri',25,True,False)
clock = pygame.time.Clock()

posRect= [750, 250, 20, 20]
posRect0= [250, 750, 10, 10]
x=0
y=0
x0=0
y0=0
score=0
timeRemain=600


poshuman = [750, 250, 20, 20]
posdemon = [250, 750, 10, 10]
int pointFdemon = 0
int pointFhuman = 0

def evalFhuman(a):
    pointFhuman += a
def evalFdemon(a):
    pointFdemon += a
def reword(c,d):
    if poshuman = posdemon:
        evalFhuman(1)
        evalFdemon(-1)

def punish(e,f):
    if timeRemain:
        evalFhuman(-1)
        evalFdemon(1)
while not done:

    screen.fill((255,255,255))

    for event in pygame.event.get():
        if event.key == K_ESCAPE:
            done = True

        if event.key == :
            u , v = j.get_axis(0), j.get_axis(1)
            if int(u*10)+int(v*10) != 0:
                x= x + int(u*10)
                y= y + int(v*10)
                if x > :
                    x= 100
                if x < -100:
                    x= -100
                if y > 80:
                    y= 80
                if y < -80:
                    y= -80
                poshuman = [750+x, 250+y, 20, 20]

    r= numpy.random.rand(2)
    speed= 0.5
    x0= x0 + int(r[0]*20-10)*speed
    y0= y0 + int(r[1]*20-10)*speed
    if x0 > 1000:
        x0= 1000
    if x0 < -1000:
        x0= -1000
    if y0 > 1000:
        y0= 1000
    if y0 < -1000:
        y0= -1000
    posRect0= [120-5+x0, 90-5+y0, 10, 10]

    if (x0 - x)**2 + (y0 - y)**2 < 150:
        score= score + 1

    pygame.draw.rect(screen, (255,0,0), posRect, 2)
    pygame.draw.rect(screen, (0,0,255), posRect0, 2)

    timeRemain= timeRemain - 1
    text= font.render(str(score)+' '+str(timeRemain),True,(0,0,0))
    screen.blit(text,[0,0])
    pygame.display.flip()

    if timeRemain == 0:
        pygame.time.wait(10000)
        done = True

    clock.tick(10)

pygame.quit()


# float getRadian(float X1,float Y1,float X2,float Y2){
#     float w = X2 - X1; # cosθ
#     float h = Y2 - Y1; # sinθ
#
#     if( w != 0 ){
#         float t_Work = h / w; # tanθ
#         if (X1 < X2) {
#             # ターゲットX座標が大きい場合はそのまま
#             return atanf(t_Work);
#         }
#         # ターゲットX座標が小さい場合は反対に行ってしまうため180度足す
#         return atanf(t_Work) + 3.1415f;
#     }
#
#     # 以下ターゲットが同一X座標の場合の処理
#
#     if (Y1 < Y2){
#         # ターゲットY座標が大きい場合は真上なので90度
#         return 3.1415f / 2.0f;
#     }
#
#     # ターゲットY座標が小さい場合は真下なので-90度(もしくは270度)
#     return -3.1415f / 2.0f;
# }

float getRadian(float X1,float Y1,float X2,float Y2){
    float w = X2 - X1; # cosθ
    float h = Y2 - Y1; # sinθ

    return atan2f(h,w);
}
