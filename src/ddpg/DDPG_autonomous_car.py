#!/usr/bin/python

import tensorflow as tf
import numpy as np
import os
import shutil
import time
import rospy
import sys
import seaborn as sns
import matplotlib.pyplot as plt

from autonomous_car import Car

np.random.seed(1995)
tf.set_random_seed(1995)

MAX_EPISODES = 50000
MAX_EP_STEPS = 250
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-3  # learning rate for critic
GAMMA = 0.99  # reward discount
REPLACE_ITER_A = 2000
REPLACE_ITER_C = 2000
MEMORY_CAPACITY = 1000000
LEARNING_START_RATIO = float(1)/4
BATCH_SIZE = 64
VAR_MIN = 0.01
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
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 400, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys= -a_grads) 

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(learning_rate=self.lr) # negative gradient for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)

            with tf.variable_scope('l1'):
                n_l1 = 100
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 400, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1


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
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

saver = tf.train.Saver()
path = './model'#+MODE[n_model]

if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    sess.run(tf.global_variables_initializer())

reward_hist = []
avg_reward_hist = []


def displayRewardHist():

    plt.figure(1)
    sns.set(style="darkgrid")
    plt.plot(reward_hist, label='Reward History')
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


avg_reward = 0
def train():
    global avg_reward, reward_hist, avg_reward_hist

    var = 1.00  # control exploration

    ep = 1
    while ep < MAX_EPISODES and not rospy.is_shutdown():
        s = env.reset()
        ep_reward = 0

        t = 0
        while t < MAX_EP_STEPS and not rospy.is_shutdown():

            try:
                # Added exploration noise
                a = actor.choose_action(s)
                a = np.clip(np.random.normal(a, var), *ACTION_BOUND)    # add randomness to action selection for exploration
                s_, r, done, reason = env.step(s,a,t)
                M.store_transition(s, a, r, s_)

                if M.pointer > (MEMORY_CAPACITY):  #* LEARNING_START_RATIO
                    var = max([var*.999999, VAR_MIN])    # decay the action randomness
                    b_M = M.sample(BATCH_SIZE)
                    b_s = b_M[:, :STATE_DIM]
                    b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                    b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                    b_s_ = b_M[:, -STATE_DIM:]

                    critic.learn(b_s, b_a, b_r, b_s_)
                    actor.learn(b_s)

                s = s_
                ep_reward += r

                if t == MAX_EP_STEPS-1 or done:
                # if done:
                    result = '| done' if done else '| ----'
                    print('Ep:', ep,
                          result,
                          '| R: %.2f' % ep_reward,
                          '| Explore: %.2f' % var,
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
    ckpt_path = os.path.join(path, 'DDPG.ckpt')
    save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    print("\nSave Model %s\n" % save_path)

    displayRewardHist()


# def eval():
#     env.set_fps(30)
#     s = env.reset()
#     while True:
#         if RENDER:
#             env.render()
#         a = actor.choose_action(s)
#         s_, r, done = env.step(a)
#         s = s_

if __name__ == '__main__':

    env.initializeTimings()

    if LOAD:
        eval()
    else:
        train()

    rospy.spin()