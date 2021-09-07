import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym
from tqdm import tqdm

# Define some Hyper Parameters
BATCH_SIZE = 32     # batch size of sampling process from buffer
LR = 0.01           # learning rate
EPSILON = 0.1       # epsilon used for epsilon greedy approach
GAMMA = 0.9         # discount factor
TARGET_NETWORK_REPLACE_FREQ = 100       # How frequently target netowrk updates
MEMORY_CAPACITY = 2048               # The capacity of experience replay buffer
EPOCH = 400
LEARNING_STEP = 4000
# GAME_NAME = "CartPole-v0"
GAME_NAME = "MountainCar-v0"


env = gym.make(GAME_NAME)  # Use cartpole game as environment
env = env.unwrapped
N_ACTIONS = env.action_space.n  # 2 actions
N_STATES = env.observation_space.shape[0]  # 4 states
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(
), int) else env.action_space.sample().shape     # to confirm the shape


class Net(nn.Module):
    def __init__(self):
        # Define the network structure, a very simple fully connected network
        super(Net, self).__init__()
        # Define the structure of fully connected network
        self.fc1 = nn.Linear(N_STATES, 10)  # layer 1
        # in-place initilization of weights of fc1
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, N_ACTIONS)  # layer 2
        # in-place initilization of weights of fc2
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # Define how the input data pass inside the network
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

# 3. Define the DQN network and its corresponding methods


class DQN(object):
    def __init__(self):
        # -----------Define 2 networks (target and training)------#
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0  # count the steps of learning process
        self.memory_counter = 0  # counter used for experience replay buffer

        # ----Define the memory (or the buffer), allocate some space to it. The number
        # of columns depends on 4 elements, s, a, r, s_, the total is N_STATES*2 + 2---#
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x, explore=True):
        # This function is used to make decision based upon epsilon greedy

        # add 1 dimension to input state x
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # action = None
        # input only one sample
        if explore == True:
            e = 0.01
        else:
            e = EPSILON
        if np.random.uniform() < e:
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(
                ENV_A_SHAPE)
            return action

        actions_value = self.eval_net.forward(x)
        action = torch.max(actions_value, 1)[1].data.numpy()
        action = action[0] if ENV_A_SHAPE == 0 else action.reshape(
            ENV_A_SHAPE)  # return the argmax index

        return action

    def store_transition(self, s, a, r, s_):
        # This function acts as experience replay buffer
        # horizontally stack these vectors
        transition = np.hstack((s, [a, r], s_))
        # if the capacity is full, then use index to replace the old memory with new one
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # Define how the whole DQN works including sampling batch of experiences,
        # when and how to update parameters of target network, and how to implement
        # backward propagation.

        # update the target network every fixed steps
        if self.learn_step_counter % TARGET_NETWORK_REPLACE_FREQ == 0:
            # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # Determine the index of Sampled batch from buffer
        # randomly select some data from buffer
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        # extract experiences of batch size from buffer.
        b_memory = self.memory[sample_index, :]
        # extract vectors or matrices s,a,r,s_ from batch memory and convert these to torch Variables
        # that are convenient to back propagation
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        # convert long int type to tensor
        b_a = Variable(torch.LongTensor(
            b_memory[:, N_STATES:N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        # calculate the Q value of state-action pair
        q_eval = self.eval_net(b_s).gather(1, b_a)  # (batch_size, 1)
        # print(q_eval)
        # calculate the q value of next state
        # detach from computational graph, don't back propagate
        q_next = self.target_net(b_s_).detach()
        # select the maximum q value
        # print(q_next)
        # q_next.max(1) returns the max value along the axis=1 and its corresponding index
        q_target = b_r + GAMMA * \
            q_next.max(1)[0].view(BATCH_SIZE, 1)  # (batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()  # reset the gradient to zero
        loss.backward()
        self.optimizer.step()  # execute back propagation for one step


def play_montecarlo(env, agent, render=False, explore=True):
    episode_reward = 0.  # 记录回合总奖励，初始化为0
    observation = env.reset()  # 重置游戏环境，开始新回合
    for _ in range(LEARNING_STEP):  # 不断循环，直到回合结束，没有结束就跳出循环
        if render:  # 判断是否显示
            env.render()  # 显示图形界面，图形界面可以用 env.close() 语句关闭
        action = agent.choose_action(observation, explore)
        next_observation, reward, done, _ = env.step(action)  # 执行动作
        episode_reward += reward  # 收集回合奖励
        if explore:  # 判断是否训练智能体
            agent.learn(observation, action, reward, done)  # 学习
        if done:  # 回合结束，跳出循环
            break
        observation = next_observation
    return episode_reward  # 返回回合总奖励


'''
--------------Procedures of DQN Algorithm------------------
'''
# create the object of DQN class
dqn = DQN()
r_log = []

# Start training
print("\nCollecting experience...")
for i_episode in tqdm(range(EPOCH)):
    # play 400 episodes of cartpole game
    s = env.reset()
    ep_r = 0
    for step in range(LEARNING_STEP):
        # take action based on the current state
        a = dqn.choose_action(s)
        # obtain the reward and next state and some other information
        s_, r, done, info = env.step(a)
        if done:
            r += 100

        # 奖励函数（非常重要，影响收敛结果和收敛速度）
        position, _ = s_
        # r = abs(position - (-0.5))
        if position > -0.2:
            r += 0.2
        elif position > -0.15:
            r += 0.5
        elif position > -0.1:
            r += 0.7

        # x, x_dot, theta, theta_dot = s_
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta)) / \
        #     env.theta_threshold_radians - 0.5
        # reward = r1 + r2

        # store the transitions of states
        dqn.store_transition(s, a, r, s_)

        ep_r += r
        # if the experience repaly buffer is filled, DQN begins to learn or update
        # its parameters.
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            # if done:
            #     print('Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r, 2))

        if done:
            # if game is over, then skip the while loop.
            break
        # use next state to update the current state.
        s = s_
    episode_rewards = [play_montecarlo(
        env, dqn, explore=False) for _ in range(100)]
    # episode_rewards = [play_montecarlo(env, dqn) for _ in tqdm(range(10))]
    r_log.append(np.mean(episode_rewards))

# save and display reward_log
np.save("{}_r_log_epoch_{}_{}.npy".format(GAME_NAME, EPOCH),
        r_log, LEARNING_STEP)
np.save("{}_dqn{}_{}.npy".format(GAME_NAME, EPOCH),
        dqn, LEARNING_STEP)

xpoints = range(EPOCH)
ypoints = r_log[:]

plt.plot(xpoints, ypoints)
plt.show()
