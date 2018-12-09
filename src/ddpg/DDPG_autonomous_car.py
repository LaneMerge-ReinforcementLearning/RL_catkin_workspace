#!/usr/bin/python

import tensorflow as tf
import numpy as np
import os
import shutil
import pygame
import rospy
import sys
import seaborn as sns
import matplotlib.pyplot as plt

from autonomous_car import Car
from OU_Noise import OUNoise

np.random.seed(1)
tf.set_random_seed(1)

NOISY_EPISODES = 2000
MAX_EPISODES = 2000
MAX_EP_STEPS = 10000
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-3  # learning rate for critic
GAMMA = 0.9  # reward discount
MEMORY_CAPACITY = 5000
BATCH_SIZE = 64
RENDER = False
LOAD = False
TAU = 0.001
# MODE = ['easy', 'hard']
# n_model = 1

env = Car()
STATE_DIM = env.state_dim
ACTION_DIM = env.action_dim
ACTION_BOUND = env.action_bound

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, shape=[None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, s, s_):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate

        self.s = s
        self.s_ = s_

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(self.s)

            self.e_params = tf.trainable_variables()
            ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
            self.target_update = ema.apply(self.e_params)

            self.a_ = self._build_net(self.s_, reuse=True, getter=self.get_getter(ema))


    def get_getter(self, ema):
        def ema_getter(getter, name, *args, **kwargs):
            var = getter(name, *args, **kwargs)
            ema_var = ema.average(var)
            return ema_var if ema_var else var

        return ema_getter

    def _build_net(self, s, reuse=None, getter=None):
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=getter):
            init_w = tf.random_normal_initializer(0.0, 0.03)
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 400, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1')
            net = tf.layers.dense(net, 300, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2')
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          name='a')
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s, s_):   # batch update
        self.sess.run([self.train_op, self.target_update], feed_dict={self.s: s, self.s_:s_})

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={self.s: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(learning_rate = -self.lr) # negative gradient for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, a, a_, s, r, s_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma

        self.s = s
        self.r = r
        self.s_ = s_

        # Input (s, a), output q
        self.a = a
        self.a_ = a_
        self.q = self._build_net(self.s, self.a)

        self.e_params = tf.trainable_variables()
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        self.target_update = ema.apply(self.e_params)

        # Input (s_, a_), output q_ for q_target
        self.q_ = self._build_net(self.s_, self.a_, reuse=True, getter=self.get_getter(ema))


        with tf.variable_scope('target_q'):
            self.target_q = self.r + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.squared_difference(self.target_q, self.q)

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, self.a)[0]   # tensor of gradients of each sample (None, a_dim)

    def get_getter(self, ema):
        def ema_getter(getter, name, *args, **kwargs):
            var = getter(name, *args, **kwargs)
            ema_var = ema.average(var)
            return ema_var if ema_var else var

        return ema_getter

    def _build_net(self, s, a, reuse=None, getter=None):
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=getter):
            init_w = tf.random_normal_initializer(0.0, 0.03)
            init_b = tf.constant_initializer(0.001)

            n_l1 = 400
            net = tf.layers.dense(s, n_l1, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1')
            with tf.variable_scope('l2'):
                n_l2 = 300
                w2_s = tf.get_variable('w2_s', [n_l1, n_l2], initializer=init_w)
                w2_a = tf.get_variable('w2_a', [self.a_dim, n_l2], initializer=init_w)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=init_b)
                net = tf.nn.relu(tf.matmul(net, w2_s) + tf.matmul(a, w2_a) + b2)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run([self.train_op, self.target_update], feed_dict={self.s: s, self.a: a, self.r: r, self.s_: s_})


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= (self.capacity), 'Memory has not been fulfilled' #* LEARNING_START_RATIO
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


sess = tf.Session()

# Create actor and critic.
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, S, S_)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, actor.a, actor.a_, S, R, S_)
actor.add_grad_to_graph(critic.a_grads)

M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

exploration_noise = OUNoise(ACTION_DIM)

saver = tf.train.Saver()
path = './model/'

if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    sess.run(tf.global_variables_initializer())

reward_hist = []
avg_reward_hist = []


def displayRewardHist():

    plt.figure(1)
    sns.set(style="darkgrid")
    plt.plot(reward_hist, lw=1, label='Reward History')
    plt.xlabel('Episode')
    plt.ylabel('Epsode Reward')
    plt.legend(loc='best')

    plt.figure(2)
    sns.set(style="darkgrid")
    plt.plot(avg_reward_hist, label='Average Reward trend')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend(loc='best')

    plt.show()


accel, steer = 0.0, 0.0
human_input = False
def addHumanSupervision(action):

    global accel, steer, human_input

    for event in pygame.event.get():

        if event.type == pygame.KEYDOWN:
            human_input = True

            if event.key == pygame.K_UP:
                accel = ACTION_BOUND[1]
            elif event.key == pygame.K_DOWN:
                accel = ACTION_BOUND[0]
            elif event.key == pygame.K_LEFT:
                steer = ACTION_BOUND[1]
            elif event.key == pygame.K_RIGHT:
                steer = ACTION_BOUND[0]

        if event.type == pygame.KEYUP:
            human_input = False

            if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                accel = 0.0
            elif event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                steer = 0.0

        pygame.display.update()

    if human_input:
        action[0] = steer
        action[1] = accel

    pygame.display.update()

avg_reward = 0
def train():
    global avg_reward, reward_hist, avg_reward_hist
    global steer, accel

    state = "EXPLORING"
    ep = 1
    while ep < MAX_EPISODES and not rospy.is_shutdown():
        s = env.reset()
        ep_reward = 0

        t = 0
        while t < MAX_EP_STEPS and not rospy.is_shutdown():

            try:
                # Added exploration noise
                a = actor.choose_action(s)
                if ep < NOISY_EPISODES and ep % 10 != 0:
                    a = np.clip(a+exploration_noise.noise(), *ACTION_BOUND) # add noise to action for exploration

                addHumanSupervision(a)

                s_, r, done, reason = env.step(s, a, t)
                M.store_transition(s, a, r, s_)

                if M.pointer > (MEMORY_CAPACITY):  #* LEARNING_START_RATIO
                    state = "LEARNING"

                    b_M = M.sample(BATCH_SIZE)
                    b_s = b_M[:, :STATE_DIM]
                    b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                    b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                    b_s_ = b_M[:, -STATE_DIM:]

                    critic.learn(b_s, b_a, b_r, b_s_)
                    actor.learn(b_s, b_s_)#, critic.a_grads)

                s = s_
                ep_reward += r

                if t == MAX_EP_STEPS-1 or done:
                    result = '| done' if done else '| ----'
                    print('Ep:', ep,
                          '|', state,
                          result,
                          '| R: %.2f' % ep_reward,
                          '| Pos X: %.2f' % s[0],
                          '| Pos Y: %.2f' % s[1],
                          '| Count: %d' % Car.counter,
                          '| Step: %d' % t,
                          '| Reason:', reason
                          )

                    reward_hist.append(ep_reward)
                
                    avg_reward = float(avg_reward*(ep-1) + ep_reward)/ep
                    avg_reward_hist.append(avg_reward)

                    break

                t += 1

            except KeyboardInterrupt:

                print("CAATCHEEEDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
                rospy.signal_shutdown("Keyboard Interrupt")
                sys.exit(0)

        ep += 1

    if os.path.isdir(path): shutil.rmtree(path)
    os.mkdir(path)
    ckpt_path = os.path.join(path, 'DDPG')
    save_path = saver.save(sess, ckpt_path)
    print("\nSave Model %s\n" % save_path)

    displayRewardHist()

def eval():
    done = False

    s = env.reset()
    while True and not rospy.is_shutdown():
        if done:
            s = env.reset()
        a = actor.choose_action(s)
        s_, r, done, result = env.step(s, a, -1)
        s = s_

if __name__ == '__main__':

    pygame.init()
    screen = pygame.display.set_mode((1,1))

    env.initializeTimings()

    if LOAD:
        eval()
    else:
        train()

    rospy.spin()