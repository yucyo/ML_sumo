import math
import sys
import os
import random

import numpy as np
import pygame
from pygame.locals import*

import tensorflow as tf
import gym

START, PLAY, GAMEOVER, STAGECLEAR = (0, 1, 2, 3)  # Game状態
DemonRect=Rect(0, 0, 560, 480)
HumanRect=Rect(0, 0, 560, 480)

class DemonHuman:
    def __init__(self,step = False,image = False):
        self.score = 0  # score(捕まえると100点)
        self.counter = 0  # タイムカウンター(60カウント = 1秒)
        self.step_flag = step
        self.image_flag = image
        self.observation_space = None
        pygame.init()
        self.screen = pygame.display.set_mode(HumanRect.size, pygame.DOUBLEBUF)
        pygame.display.set_caption("DemonHuman")
        # オブジェクト初期化
        self.init_game()
        self._get_observation()
        # メインループ開始
        if not self.step_flag: # 通常モード
            clock = pygame.time.Clock()
            while True:
                clock.tick(60)
                self.update()
                self.draw(self.screen)
                pygame.display.update()
                self.key_handler()
        else: # step実行時はタイトル画面を飛ばす
            self.game_state = PLAY

    # gym互換関数群---------------------ここから
    def reset(self):
        self.score = 0  # スコア初期化
        self.counter = 0
        self.no_touch_counter = 0  # humanを捕まえられていない時間（60カウント=1秒）
        self.init_game()  # ゲームを初期化して再開
        self.game_state = PLAY
        self._get_observation()
        return self.observation_space

    def render(self, mode=None):
        pass

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
        self.no_touch_counter += 1

        if (self.player.rect.center[0] - self.human.rect.center[0])**2 < 1000:
            reward += 10000
        elif self.player.rect.center[0] - self.player.rect.center[1]:
            reward += -10

    def _key_action(self, action): # キー入力を代替
        # 左移動
        if action == 1:
            self.player.rect.move_ip(-self.player.speed, 0)
            self.player.rect.clamp_ip(PLAYER_MOVE_RECT)
        # 右移動
        elif action == 2:
            self.player.rect.move_ip(self.player.speed, 0)
            self.player.rect.clamp_ip(PLAYER_MOVE_RECT)
        elif action == 3:
            self.player.rect.move_ip(self.player.speed, 0)
            self.player.rect.clamp_ip(PLAYER_MOVE_RECT)
        elif action == 4:
            self.player.rect.move_ip(-self.player.speed, 0)
            self.player.rect.clamp_ip(PLAYER_MOVE_RECT)
        pass

    def _get_observation(self):
        if self.image_flag:
            resize_x = 160
            resize_y = 120
            cut_y_rate = 0.06
            pilImg = Image.fromarray(pygame.surfarray.array3d(self.screen))
            resizedImg = pilImg.resize((resize_x, resize_y), Image.LANCZOS)
            self.observation_space = np.asarray(resizedImg)[:][int(resize_y*cut_y_rate):]
            return None
        observation_list = list()
        for index, human in enumerate(self.human_list):
            if human.alive:
                observation_list.append((human.rect.center[0] - self.player.rect.center[0])/640)
                observation_list.append((human.rect.center[1] - self.player.rect.center[1])/480)
                observation_list.append(human.speed/2)
            else:
                observation_list.extend([0, 0, 0])
        observation_list.append(len(self.human_list))

        pass

    #gym互換関数群--------------ここまで

    def init_game(self):
        """ゲームオブジェクトを初期化"""
        # ゲーム状態
        self.game_state = START
        # スプライトグループを作成して登録
        self.all = pygame.sprite.RenderUpdates()
        self.invisible = pygame.sprite.RenderUpdates()
        self.human = pygame.sprite.Group()  # humanグループ
        # self.demon = pygame.sprite.Group()   # demonグループ
        # デフォルトスプライトグループを登録
        Player.containers = self.all
        # demon.containers = self.all, self.demon, self.invisible
        # 自分(demon)を作成
        self.player = Player()
        # humanを作成
        self.alien_list = list()
        for i in range(0, 50):
            x = 10 + (i % 10) * 40
            y = 50 + (i // 10) * 40
            self.alien_list.append(Alien((x,y), self.wave))
        # # 壁を作成
        # '''
        # self.wall_list = list()
        # for i in range(4):
        #     x = 95 + i * 150
        #     y = 400
        #     self.wall_list.append(Wall((x, y), self.wave))
        # '''

def update(self):
    """ゲーム状態の更新"""
    if self.game_state == PLAY:
        # タイムカウンターを進める
        self.counter += 1
        self.all.update()
        # エイリアンの方向転換判定
        turn_flag = False
        for human in self.human:
            if (human.rect.center[0] < 10 and alien.speed < 0) or \
                    (human.rect.center[0] > human.width-10 and human.speed > 0):
                turn_flag = True
                break
        if turn_flag:
            for human in self.human:
                human.speed *= -1
        # ミサイルとエイリアン、壁の衝突判定
        self.collision_detection()
        # スコアを1000とったらクリア
        if self.score == 1000:
            self.game_state = STAGECLEAR

    def draw(self, screen):
        """描画"""
        screen.fill((0, 0, 0))
        if self.game_state == START:  # スタート画面
            # タイトルを描画
            title_font = pygame.font.SysFont(None, 80)
            title = title_font.render("DemonHuman", False, (255,0,0))
            screen.blit(title, ((HumanRect.width-title.get_width())//2, 100))
            # Humanを描画
            human_image = human.images[0]
            screen.blit(human_image, ((HumanRect.width-human_image.get_width())//2, 200))
            # PUSH STARTを描画
            push_font = pygame.font.SysFont(None, 40)
            push_space = push_font.render("PUSH SPACE KEY", False, (255,255,255))
            screen.blit(push_space, ((HumanRect.width-push_space.get_width())//2, 300))
            # クレジットを描画
            credit_font = pygame.font.SysFont(None, 20)
            credit = credit_font.render("made by Osu", False, (255,255,255))
            screen.blit(credit, ((HumanRect.width-credit.get_width())//2, 380))
        elif self.game_state == PLAY:  # ゲームプレイ画面
            # wave数と残機数を描画
            stat_font = pygame.font.SysFont(None, 20)
            stat = stat_font.render("Wave:{:2d}  Lives:{:2d}  Score:{:05d}  pos_X:{:3d}".format(
                                        self.score,
                                        self.player.rect.center[0]), False, (255,255,255))
            screen.blit(stat, ((HumanRect.width - stat.get_width()) // 2, 10))
            # 壁の耐久力描画
            # shield_font = pygame.font.SysFont(None, 30)
            # for wall in self.walls:
            #     shield = shield_font.render(str(wall.shield), False, (0,0,0))
            #     text_size = shield_font.size(str(wall.shield))
            #     screen.blit(shield, (wall.rect.center[0]-text_size[0]//2,
            #                          wall.rect.center[1]-text_size[1]//2))
        elif self.game_state == GAMEOVER:  # ゲームオーバー画面
            # GAME OVERを描画
            gameover_font = pygame.font.SysFont(None, 80)
            gameover = gameover_font.render("GAME OVER", False, (255,0,0))
            screen.blit(gameover, ((HumanRect.width-gameover.get_width())//2, 100))
            # Humanを描画
            human_image = human.images[0]
            screen.blit(human_image, ((SCR_RECT.width-alien_image.get_width())//2, 200))
            # PUSH SPACEを描画
            push_font = pygame.font.SysFont(None, 40)
            push_space = push_font.render("PUSH SPACE KEY", False, (255,255,255))
            screen.blit(push_space, ((HumanRect.width-push_space.get_width())//2, 300))

        elif self.game_state == STAGECLEAR:  # ステージクリア画面
            # wave数と残機数を描画
            stat_font = pygame.font.SysFont(None, 20)
            stat = stat_font.render("Wave:{:2d}  Lives:{:2d}  Score:{:05d}".format(
                self.score), False, (255,255,255))
            screen.blit(stat, ((HumanRect.width - stat.get_width()) // 2, 10))
            # STAGE CLEARを描画
            gameover_font = pygame.font.SysFont(None, 80)
            gameover = gameover_font.render("STAGE CLEAR", False, (255,0,0))
            screen.blit(gameover, ((HumanRect.width-gameover.get_width())//2, 100))
            # エイリアンを描画
            human_image = human.images[0]
            screen.blit(human_image, ((HumanRect.width-alien_image.get_width())//2, 200))
            # PUSH SPACEを描画
            push_font = pygame.font.SysFont(None, 40)
            push_space = push_font.render("PUSH SPACE KEY", False, (255,255,255))
            screen.blit(push_space, ((HumanRect.width-push_space.get_width())//2, 300))

    def key_handler(self):
        """キーハンドラー"""
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN and event.key == K_SPACE:
                if self.game_state == START:  # スタート画面でスペースを押したとき
                    self.game_state = PLAY
                elif self.game_state == GAMEOVER:  # ゲームオーバー画面でスペースを押したとき
                    self.score = 0  # スコア初期化
                    self.counter = 0
                    self.init_game()  # ゲームを初期化して再開
                    self.game_state = PLAY
                elif self.game_state == STAGECLEAR:
                    self.counter = 0
                    self.init_game()  # ゲームを初期化して再開
                    self.game_state = PLAY

    def collision_detection(self):
        """衝突判定"""
        # # エイリアンとミサイルの衝突判定
        # alien_collided = pygame.sprite.groupcollide(self.aliens, self.shots, True, True)
        # for alien in alien_collided.keys():
        #     Alien.kill_sound.play()
        #     self.score += 10 * self.wave
        #     Explosion(alien.rect.center)  # エイリアンの中心で爆発
        # # UFOとミサイルの衝突判定
        # ufo_collided = pygame.sprite.groupcollide(self.ufos, self.shots, True, True)
        # for ufo in ufo_collided.keys():
        #     Alien.kill_sound.play()
        #     self.score += 50 * self.wave
        #     Explosion(ufo.rect.center)
        #     self.lives += 1
        # # プレイヤーとビームの衝突判定
        # # 無敵時間中なら判定せずに無敵時間を1減らす
        # if self.player.invisible > 0:
        #     beam_collided = False
        #     self.player.invisible -= 1
        # else:
        #     beam_collided = pygame.sprite.spritecollide(self.player, self.beams, True)
        # if beam_collided:  # プレイヤーと衝突したビームがあれば
        #     Player.bomb_sound.play()
        #     Explosion(self.player.rect.center)
        #     self.lives -= 1
        #     self.player.invisible = 0 # DQN用は無敵時間無し
        #     if self.lives < 0:
        #         self.game_state = GAMEOVER  # ゲームオーバー！
        # # 壁とミサイル、ビームの衝突判定
        # hit_dict = pygame.sprite.groupcollide(self.walls, self.shots, False, True)
        # hit_dict.update(pygame.sprite.groupcollide(self.walls, self.beams, False, True))
        # for hit_wall in hit_dict:
        #     hit_wall.shield -= len(hit_dict[hit_wall])
        #     for hit_beam in hit_dict[hit_wall]:
        #         Alien.kill_sound.play()
        #         Explosion(hit_beam.rect.center)  # ミサイル・ビームの当たった場所で爆発
        #     if hit_wall.shield <= 0:
        #         hit_wall.kill()
        #         Alien.kill_sound.play()
        #         ExplosionWall(hit_wall.rect.center)  # 壁の中心で爆発
        pass

    def load_images(self):
        """イメージのロード"""
        # スプライトの画像を登録
        player.image = load_image("player.png")
        human.images = split_image(load_image("alien.png"), 2)
        UFO.images = split_image(load_image("ufo.png"), 2)
        Beam.image = load_image("beam.png")
        # Wall.image = load_image("wall.png")
        # Explosion.images = split_image(load_image("explosion.png"), 16)
        # ExplosionWall.images = split_image(load_image("explosion2.png"), 16)

    def load_sounds(self):
        """サウンドのロード"""
        Alien.kill_sound = load_sound("kill.wav")
        Player.shot_sound = load_sound("shot.wav")
        Player.bomb_sound = load_sound("bomb.wav")

class Player(pygame.sprite.Sprite):
    """自機"""
    speed = 5  # 移動速度
    invisible = 0  # 無敵時間
    def __init__(self):
        # imageとcontainersはmain()でセット
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.rect = self.image.get_rect()
        self.rect.center = (HumanRect.width//2, HumanRect.bottom - 9)
        self.reload_timer = 0
    def update(self):
        # 押されているキーをチェック
        pressed_keys = pygame.key.get_pressed()
        # 押されているキーに応じてプレイヤーを移動
        if pressed_keys[K_LEFT]:
            self.rect.move_ip(-self.speed, 0)
        elif pressed_keys[K_RIGHT]:
            self.rect.move_ip(self.speed, 0)
        #self.rect.clamp_ip(SCR_RECT)
        self.rect.clamp_ip(DemonRect)
        # ミサイルの発射
        # if pressed_keys[K_SPACE]:
        #     # リロード時間が0になるまで再発射できない
        #     '''
        #     if self.reload_timer > 0:
        #         #
        #         self.reload_timer -= 1
        #     '''
        #     if self.reload_timer == 0:
        #         # 発射！！！
        #         Player.shot_sound.play()
        #         Shot(self.rect.center)  # 作成すると同時にallに追加される
        #         self.reload_timer = self.reload_time
        pass

class Player(pygame.sprite.Sprite):
    """エイリアン"""
    def __init__(self):
        self.speed = 2  # 移動速度
        self.animcycle = 18  # アニメーション速度
        self.frame = 0
        # self.prob_beam = (1.5 + wave) * 0.002  # ビームを発射する確率
        # imagesとcontainersはmain()でセット
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = self.images[0]
        self.rect = self.image.get_rect()
        # self.rect.center = pos

    def update(self):
        # 横方向への移動
        self.rect.move_ip(self.speed, 0)
        self.rect.move_ip(self.speed, 2)
        # キャラクターアニメーション
        self.frame += 1
        self.image = self.images[self.frame//self.animcycle%2]


def load_image(filename, colorkey=None):
    """画像をロードして画像と矩形を返す"""
    filename = os.path.join("data", filename)
    try:
        image = pygame.image.load(filename)
    except pygame.error as message:
        print("Cannot load image:", filename)
        raise SystemExit(message)
    image = image.convert()
    if colorkey is not None:
        if colorkey is -1:
            colorkey = image.get_at((0,0))
        image.set_colorkey(colorkey, RLEACCEL)
    return image


def split_image(image, n):
    """横に長いイメージを同じ大きさのn枚のイメージに分割
    分割したイメージを格納したリストを返す"""
    image_list = []
    w = image.get_width()
    h = image.get_height()
    w1 = w // n
    for i in range(0, w, w1):
        surface = pygame.Surface((w1,h))
        surface.blit(image, (0,0), (i,0,w1,h))
        surface.set_colorkey(surface.get_at((0,0)), RLEACCEL)
        surface.convert()
        image_list.append(surface)
    return image_list


def load_sound(filename):
    """サウンドをロード"""
    filename = os.path.join("data", filename)
    return pygame.mixer.Sound(filename)


if __name__ == "__main__":
    DemonHuman()






# screen = pygame.display.set_mode((1000,1000))
# pygame.display.set_caption("TestSeeking")
# font=pygame.font.SysFont('Calibri',25,True,False)
# clock = pygame.time.Clock()
#
#
# dx=400
# dy=400
# hx=0
# hy=0
# score=0
# timeRemain=600
# Oni=[int(dx),int(dy),30,30]
# Hito=[100,100,30,30]
#
# done = False
# while not done:
#
#     screen.fill((255,255,255))
#
#     for event in pygame.event.get():
#         if event.type == pygame.quit:
#             done = True
#         # if event.key == K_ESCAPE:
#         #     done = True
#         if event.type == KEYDOWN:
#             if event.key == K_LEFT:
#                 dx -= 15
#             if event.key == K_RIGHT:
#                 dx += 15
#             if event.key == K_UP:
#                 dy -= 15
#             if event.key == K_DOWN:
#                 dy += 15
#             Oni=[dx,dy,30,30]
#             # if event.type == K_:
#         #     u , v = j.get_axis(0), j.get_axis(1)
#         #     if int(u*10)+int(v*10) != 0:
#         #         x= x + int(u*10)
#         #         y= y + int(v*10)
#         #         if x > 100:
#         #             x= 100
#         #         if x < -100:
#         #             x= -100
#         #         if y > 80:
#         #             y= 80
#         #         if y < -80:
#         #             y= -80
#         #         posRect= [120-10+x, 90-10+y, 20, 20]
#
#
#     r= np.random.rand(2)
#     speed= 1.0
#     hx= hx + int(r[0]*20-10)*speed
#     hy= hy + int(r[1]*20-10)*speed
#     if hx > 1000:
#         hx= 1000
#     if hx < -1000:
#         hx= -1000
#     if hy > 1000:
#         hy= 1000
#     if hy < -1000:
#         hy= -1000
#     Hito= [100+hx, 100+hy, 30, 30]
#
#     if (hx - dx)**2 + (hy - dy)**2 < 1000:
#         score = score + 1
#
#     pygame.draw.rect(screen, (255,0,0), Hito, 2)
#     pygame.draw.rect(screen, (0,0,255), Oni, 2)
#
#     timeRemain= timeRemain - 1
#     text= font.render(str(score)+' '+str(timeRemain),True,(0,0,0))
#     screen.blit(text,[0,0])
#     pygame.display.flip()
#
#     if timeRemain == 0:
#         pygame.time.wait(10000)
#         done = True
#     elif score == 10:
#         pygame.time.wait(10000)
#         done = True
#
#     clock.tick(10)
#
# pygame.quit()
