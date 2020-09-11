import pygame
import numpy

pygame.init()
screen = pygame.display.set_mode((240,180))
pygame.display.set_caption("My Game")
font=pygame.font.SysFont('Calibri',25,True,False)
clock = pygame.time.Clock()

posRect= [120-10, 90+10, 20, 20]
posRect0= [120-5, 90+5, 10, 10]
x=0
y=0
x0=0
y0=0
score=0
timeRemain=600

try:
    j = pygame.joystick.Joystick(0)
    j.init()
    print( 'Joystickの名称: ' + j.get_name())
    print( 'ボタン数 : ' + str(j.get_numbuttons()))
except pygame.error:
    print('Joystickが見つかりませんでした。')

done = False
while not done:

    screen.fill((255,255,255))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

        if event.type == pygame.JOYAXISMOTION:
            u , v = j.get_axis(0), j.get_axis(1)
            if int(u*10)+int(v*10) != 0:
                x= x + int(u*10)
                y= y + int(v*10)
                if x > 100:
                    x= 100
                if x < -100:
                    x= -100
                if y > 80:
                    y= 80
                if y < -80:
                    y= -80
                posRect= [120-10+x, 90-10+y, 20, 20]

    r= numpy.random.rand(2)
    speed= 0.5
    x0= x0 + int(r[0]*20-10)*speed
    y0= y0 + int(r[1]*20-10)*speed
    if x0 > 100:
        x0= 100
    if x0 < -100:
        x0= -100
    if y0 > 80:
        y0= 80
    if y0 < -80:
        y0= -80
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
