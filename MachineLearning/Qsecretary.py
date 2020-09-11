import sys
from os import mkdir
from os.path import exists
from sys import exc_info
import numpy as np
from enum import IntEnum
from scipy.stats import rankdata
import gym
from gym import spaces
from gym.utils import seeding
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import TrainIntervalLogger, TrainEpisodeLogger
import matplotlib.pyplot as plt


NUMS = 100  # 応募者数


class Result(IntEnum):
    non_adopt = 0
    adopt = 1


class Secretary(gym.Env):
    metadata = {
        'render.modes': ['human', 'ansi']
    }

    action_space = spaces.Discrete(2)  # 不採用／採用
    reward_range = [0., 1000.]

    def __init__(self, n=100):
        super().__init__()
        if n <= 0:
            raise ValueError('nは1以上の整数を指定してください')
        self.n = n
        self.adopted = []
        self.best_applicant = []
        self.observation_space = spaces.Box(low=0, high=self.n,
                                            shape=(3,), dtype='float32')
        self.seed()

    def __call_next(self):
        self.interviewees.append(self.applicants.pop())
        return int(rankdata(self.interviewees)[-1])

    def reset(self):
        # 初期化
        self.cnt = 1
        self.done = False
        self.applicants = [a for a in range(1, self.n + 1)]
        self.np_random.shuffle(self.applicants)
        self.best_applicant += [len(self.applicants) - np.where(np.array(self.applicants) == 1)[0][0]]
        self.adopted += [0]
        self.interviewees = []

        # 一人目
        self.interviewee = self.__call_next()
        self.observation = [self.interviewee, self.cnt, self.n]
        return self.observation

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if action == Result.adopt:
            self.adopted[-1] = self.cnt
            if self.interviewees[-1] == 1:
                self.reward = 1000.
            else:
                self.reward = 0.
            self.done = True
            self.observation = [self.interviewee, self.cnt, self.n]

        elif action == Result.non_adopt:
            if len(self.applicants) == 0:  # 全員面接した
                self.adopted[-1] = None
                self.reward = 0.
                self.done = True
                self.observation = [self.interviewee, self.cnt, self.n]
            else:
                self.reward = 0.
                self.done = False
                self.interviewee = self.__call_next()
                self.cnt += 1
                self.observation = [self.interviewee, self.cnt, self.n]

        return self.observation, self.reward, self.done, {}

    def render(self, mode='human', close=False):
        if mode == 'ansi':
            outfile = StringIO()
        elif mode == 'human':
            outfile = sys.stdout
        else:
            # just raise an exception
            super().render(mode=mode)

        s = ''
        if self.done:
            if self.reward > 0:
                s = '\nおめでとうございます！  最良の応募者を採用できました。\n'
            elif self.adopted[-1] is None:
                s = '\nあなたは誰も採用しませんでした。\n'
                s += '最良の応募者は' + str(self.best_applicant[-1]) + '人目でした。\n'
            elif self.reward <= 0:
                s = '\nあなたは' + str(self.interviewees[-1]) + '番目に優秀な応募者を採用しました。\n'
                s += '最良の応募者は' + str(self.best_applicant[-1]) + '人目でした。\n'
        else:
            s = '\n' + str(self.n) + '人中' + str(self.cnt) + '人目（暫定順位：' + str(self.interviewee) + '位）\n'

        outfile.write(s)
        return outfile

    def close(self):
        # just return
        super().close()

    def play(self):
        self.reset()
        while not self.done:
            print(self.applicants, self.interviewees)
            self.render()
            print('採用しますか？')
            self.step(self.__input())
            if self.done:
                self.render()
                break

    def __input(self):
        while True:
            print('[採用しない：0、採用する：1]')
            i = input()
            if i in ['0', '1']:
                break
            print('入力値が不正です。')
        return int(i)


class TrainIntervalLogger2(TrainIntervalLogger):
    def __init__(self, interval=10000):
        super().__init__(interval=interval)
        self.records = {}

    def on_train_begin(self, logs):
        super().on_train_begin(logs)
        self.records['interval'] = []
        self.records['episode_reward'] = []
        for metrics_name in self.metrics_names:
            self.records[metrics_name] = []

    def on_step_begin(self, step, logs):
        if self.step % self.interval == 0:
            if len(self.episode_rewards) > 0:
                self.records['interval'].append(self.step // self.interval)
                self.records['episode_reward'].append(np.mean(self.episode_rewards))
                metrics = np.array(self.metrics)
                assert metrics.shape == (self.interval, len(self.metrics_names))
                if not np.isnan(metrics).all():  # not all values are means
                    means = np.nanmean(self.metrics, axis=0)
                    assert means.shape == (len(self.metrics_names),)
                    for name, mean in zip(self.metrics_names, means):
                        self.records[name].append(mean)
        super().on_step_begin(step, logs)


class DQNSecretary:
    # 重み保存先
    weightdir = './data'
    weightfile = './data/dqn_{}_weights.h5'

    # モデルの初期化
    def __init__(self, n=100, recycle=True):
        print('モデルを作成します。')
        self.train_interval_logger = None

        # Get the environment and extract the number of actions.
        self.env = Secretary(n=n)
        self.env_name = 'secretary'
        self.weightfile = self.__class__.weightfile.format(self.env_name)
        self.nb_actions = self.env.action_space.n

        # Next, we build a very simple model.
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.nb_actions))
        self.model.add(Activation('linear'))
        #print(self.model.summary())

        # Finally, we configure and compile our agent.
        # You can use every built-in Keras optimizer and even the metrics!
        memory = SequentialMemory(limit=50000, window_length=1)
        policy = BoltzmannQPolicy(tau=1.)
        self.dqn = DQNAgent(model=self.model, nb_actions=self.nb_actions, memory=memory,
                            nb_steps_warmup=1000, target_model_update=1e-2, policy=policy)
        self.dqn.compile(Adam(lr=1e-3), metrics=[])

        self.__istrained = False
        print('モデルを作成しました。')

        if recycle:
            if exists(self.weightfile):
                try:
                    print('訓練済み重みを読み込みます。')
                    self.dqn.load_weights(self.weightfile)
                    self.__istrained = True
                    print('訓練済み重みを読み込みました。')
                    return None
                except:
                    print('訓練済み重みの読み込み中にエラーが発生しました。')
                    print('Unexpected error:', exc_info()[0])
                    raise
            else:
                print('訓練済み重みが存在しません。訓練を行ってください。')

    # 訓練
    def train(self, nb_steps=30000, verbose=1, visualize=False, log_interval=3000):
        if self.__istrained:
            raise RuntimeError('このモデルは既に訓練済みです。')

        print('訓練を行うので、お待ちください。')

        # 訓練実施
        # Okay, now it's time to learn something!
        # We visualize the training here for show, but this slows down training quite a lot.
        # You can always safely abort the training prematurely using Ctrl + C.
        callbacks = []
        if verbose == 1:
            self.train_interval_logger = TrainIntervalLogger2(interval=log_interval)
            callbacks.append(self.train_interval_logger)
            verbose = 0
        elif verbose > 1:
            callbacks.append(TrainEpisodeLogger())
            verbose = 0

        hist = self.dqn.fit(self.env, nb_steps=nb_steps,
                            callbacks=callbacks, verbose=verbose,
                            visualize=visualize, log_interval=log_interval)
        self.__istrained = True

        if self.train_interval_logger is not None:
            # 訓練状況の可視化
            interval = self.train_interval_logger.records['interval']
            episode_reward = self.train_interval_logger.records['episode_reward']
            mean_q = self.train_interval_logger.records['mean_q']
            if len(interval) > len(mean_q):
                mean_q = np.pad(mean_q, [len(interval) - len(mean_q), 0], "constant")
            plt.figure()
            plt.plot(interval, episode_reward, marker='.', label='報酬')
            plt.plot(interval, mean_q, marker='.', label='Q値')
            plt.legend(loc='best', fontsize=10)
            plt.grid()
            plt.xlabel('interval')
            plt.ylabel('score')
            plt.title('訓練状況')
            plt.xticks(np.arange(min(interval),
                                 max(interval) + 1,
                                 (max(interval) - min(interval))//7))
            plt.show()

        # 重みの保存
        if not exists(self.__class__.weightdir):
            try:
                mkdir(self.__class__.weightdir)
            except:
                print('重み保存フォルダの作成中にエラーが発生しました。')
                print('Unexpected error:', exc_info()[0])
                raise
        try:
            # After training is done, we save the final weights.
            self.dqn.save_weights(self.weightfile, overwrite=True)
        except:
            print('重みの保存中にエラーが発生しました。')
            print('Unexpected error:', exc_info()[0])
            raise

        return hist

    # テスト
    def test(self, nb_episodes=10, visualize=True, verbose=1):
        # Finally, evaluate our algorithm for 5 episodes.
        hist = self.dqn.test(self.env, nb_episodes=nb_episodes,
                             verbose=verbose, visualize=visualize)
        return hist


def visualize_adopting(env):
    plt.figure()
    x = range(1, len(env.adopted) + 1)
    plt.plot(x, env.adopted, marker='.', label='採用者')
    plt.plot(x, env.best_applicant, marker='.', alpha=0.5, label='最良の応募者')
    plt.legend(loc='best', fontsize=10)
    plt.grid()
    plt.xlabel('試行回数')
    plt.ylabel('面接者数')
    plt.ylim(0, env.n)
    plt.title('面接／採用状況（n=' + str(NUMS) + '）')
    plt.show()


def visualize_q(dqn_agent, rank):
    x = range(1, NUMS + 1)
    y = []
    z = []

    for i in x:
        y += [d.dqn.compute_q_values([[rank, i, NUMS]])[1]]
        z += [d.dqn.compute_q_values([[rank, i, NUMS]])[0]]

    mx = int(max([max(y), max(z)]))
    mn = int(min([min(y), min(z)]))

    s = [NUMS / np.e for _ in range(mn, mx)]
    t = range(mn, mx)

    plt.figure()
    plt.plot(x, y, marker='.', label='採用する')
    plt.plot(x, z, marker='.', alpha=0.5, label='採用しない')
    plt.plot(s, t, marker='.', alpha=0.5, label='x = n / e')
    plt.legend(loc='best', fontsize=10)
    plt.grid()
    plt.xlabel('人数')
    plt.ylabel('Q値')
    plt.title('Q値（n=' + str(NUMS) + '、相対順位=' + str(rank) + '）')
    plt.show()

#人がプレイする
#env = Secretary(n=10)
#env.play()

d = DQNSecretary(n=NUMS, recycle=False)
h = d.train(nb_steps=200000, log_interval=10000, verbose=1)

# d = DQNSecretary(n=NUMS, recycle=True)
# h = d.test(nb_episodes=1, verbose=1, visualize=True)
#
# h = d.test(nb_episodes=1000, visualize=False, verbose=0)
# rwds = h.history['episode_reward']
# win_rate = sum(rwd > 0 for rwd in rwds) / len(rwds)
# print('採用成功率(1000回)：' + str(win_rate))
# visualize_adopting(d.env)
#
# visualize_q(d, 1)
# for i in range(2, 10):
#     visualize_q(d, i)
