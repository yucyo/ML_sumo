from keras.models import Sequential
from keras.layers import Dense, Flatten,  Conv2D, Reshape
from keras.optimizers import Adam
from keras.regularizers import l2

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import Q_NOTchangeINVADE

env = Q_NOTchangeINVADE.Invader(step=True, image=True)
nb_actions = 3

hidden_size = 128
n_filters = 8
kernel = (13, 13)
strides = (3, 3)

model = Sequential()
model.add(Reshape((env.observation_space.shape), input_shape=(1, ) + env.observation_space.shape))
model.add(Conv2D(n_filters, kernel, strides=strides, activation='relu', padding='same'))
model.add(Conv2D(n_filters, kernel, strides=strides, activation='relu', padding='same'))
model.add(Conv2D(n_filters, kernel, strides=strides, activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(hidden_size, kernel_initializer='he_normal', activation='relu',
                kernel_regularizer=l2(0.01)))
model.add(Dense(hidden_size, kernel_initializer='he_normal', activation='relu',
                kernel_regularizer=l2(0.01)))
model.add(Dense(hidden_size, kernel_initializer='he_normal', activation='relu',
                kernel_regularizer=l2(0.01)))
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())

memory = SequentialMemory(limit=100000, window_length=1)
policy = EpsGreedyQPolicy(eps=0.001)

dqn = DQNAgent(model=model, nb_actions=nb_actions,gamma=0.99, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

fname = "invader_model_image_160x120_act3_h128_f8_k13_13_st3_c3_d3.bin"
try:
    dqn.load_weights(fname)
    print("Weights are loaded.")
except:
    print("Weights are NOT loaded.")

history = dqn.fit(env, nb_steps=200000, verbose=2)

dqn.save_weights(fname, overwrite=True)

dqn.test(env, nb_episodes=50)
