from __future__ import division
import gym
gym.logger.set_level(40)
from numpy.core._multiarray_umath import ndarray

env = gym.make("LunarLander-v2")

from collections import namedtuple, deque
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, random
import gym
import warnings
import torch.autograd as autograd
import pandas as pd

warnings.filterwarnings("ignore")
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("LunarLander-v2")
env.seed(0)
print("action space", env.action_space)  # actions: [do nothing, fire left, fire main, fire right]
print("observation spac", env.observation_space)  # Box(8, )

# let's see what a dom agent does inside the environment
env.reset()
epochs = 0
done = False
while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    # env.render()
    epochs += 1

print('Timesteps taken: {}'.format(epochs))
print('Episode ended with a reward {}, in state {}'.format(reward, state))

print("this is noisy")
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
                                                                                                                **kwargs)


# Experience replay memory for training our DQN. stores the transitions that the agent observes
class ReplayMemory(object):

    def __init__(self, action_size, capacity, batch_size, seed):
        self.capacity = capacity
        self.action_size = action_size
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        # this will add an experience to memory
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        experiences = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)


# our Noisy Network

class NoisyDQN(nn.Module):

    def __init__(self, state_size, action_size, fc1_size=128, fc2_size=128):
        super(NoisyDQN, self).__init__()
        hidden_size = 30
        self.linear = nn.Linear(state_size, fc1_size)
        self.noisy1 = NoisyLinear(fc1_size, fc2_size)
        self.noisy2 = NoisyLinear(fc2_size, action_size)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = F.relu(self.noisy1(x))
        x = self.noisy2(x)
        return x

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

# =================================================== Lunar agent ======================================================
class Lunar_agent:
    def __init__(self, state_size, action_size, seed, batch_size=64, gamma=0.99, learning_rate=1e-4,
                 capacity=int(1e5), update_every=4, tau=1e-3, pretrained = False):
        self.state_size = state_size  # Observation array length = env.observation_space.shape[0]
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.capacity = capacity
        self.update_every = update_every
        self.tau = tau
        self.Loss = 0
        self.pretrained = pretrained

        # Q-Network
        self.policy_net = NoisyDQN(self.state_size, self.action_size).to(device)
        self.target_net = NoisyDQN(self.state_size, self.action_size).to(device)
        if self.pretrained:
            self.policy_net.load_state_dict(torch.load('checkpoint_lunar_agent_non_terminal.pth'))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory(self.action_size, self.capacity, self.batch_size, self.seed)
        self.time_step = 0  # initialize time step for updating every predefined number of steps

    def step(self, state, action, reward, next_state, done):
        # save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # We are updating every UPDATE_EVERY time steps
        self.time_step = (self.time_step + 1) % self.update_every
        if self.time_step == 0:
            # if enough samples are available in memory, take a random subset
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.train(experiences, self.gamma)

    def get_action(self, state, epsilon=0):
        """
        This method will return the actions for given state following the current policy
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()

        # new
        return np.argmax(action_values.cpu().data.numpy())
        # end new

    def soft_update(self, policy_net, target_net, tau):
    
         #tau for interpolation 
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def train(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences
        # we need the max predicted Q values for next states from our target model
        Q_targets_net = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute the expected Q values
        Q_targets = rewards + (gamma * Q_targets_net * (1 - dones))  # done = 0 for False and 1 True
        # Q from policy
        Q_expected = self.policy_net(states).gather(1, (actions.type(torch.LongTensor).to(device)))
        loss = F.mse_loss(Q_expected, Q_targets)
        self.Loss = loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------------- update target network ---------------------------------- #
        self.soft_update(policy_net=self.policy_net, target_net=self.target_net, tau=self.tau)

        self.policy_net.reset_noise()
        self.target_net.reset_noise()

    def get_Loss(self):
        return self.Loss


print(30 * '#' + ' Training the agent with Q learning ' + 30 * '#')
# =================================================== Training =======================================================
lunar_agent = Lunar_agent(state_size=env.observation_space.shape[0], action_size=4, seed=0)


def train(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.9995):
    """Deep Q Learning
    :param
    n_episodes (int): maximum number of training episodes
    max_t (int): maximum number of timesteps per episode
    eps_start (float): starting value of epsilon, epsilon-greedy action selection
    eps_end (float): minimum value of epsilon
    eps_decay (float): factor for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    avg_loss_list = []
    avg_loss_window = deque(maxlen=100)
    episode_length_list = []
    episode_length_window = deque(maxlen=100)
    eps = eps_start
    start = time.time()
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        avg_loss = 0
        epis_length = 0
        for t in range(max_t):
            action = lunar_agent.get_action(state, eps)
            next_state, reward, done, _ = env.step(action)
            lunar_agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            avg_loss += float(lunar_agent.get_Loss())
            epis_length += 1
            if done:
                break
        episode_length_list.append(epis_length)
        episode_length_window.append(epis_length)
        scores_window.append(score)
        scores.append(score)
        avg_loss_window.append(avg_loss)
        avg_loss_list.append(avg_loss)
        eps = max(eps_end, eps_decay * eps)
        print('\rEpisode {}\tAverage Score: {:.2f} \taverage Loss {:.2f} \tepisode length {:.2f}'.format(i_episode,
                                                                                                         np.mean(
                                                                                                             scores_window),
                                                                                                         np.mean(
                                                                                                             avg_loss_window),
                                                                                                         np.mean(
                                                                                                             episode_length_window)),
              end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f} \taverage Loss {:.2f} \tepisode length {:.2f}'.format(i_episode,
                                                                                                             np.mean(
                                                                                                                 scores_window),
                                                                                                             np.mean(
                                                                                                                 avg_loss_window),
                                                                                                             np.mean(
                                                                                                                 episode_length_window)))
        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}.'.format(i_episode - 100,
                                                                                          np.mean(scores_window)))
            print('\ntraining took {:.2f} seconds'.format(time.time() - start))
            torch.save(lunar_agent.policy_net.state_dict(), 'checkpoint_lunar_agent.pth')
            break

    torch.save(lunar_agent.policy_net.state_dict(), 'checkpoint_lunar_agent_5000.pth')
    return scores, avg_loss_list, episode_length_list


scores, avg_loss_list, episode_length_list = train(n_episodes=5000, max_t=1000, eps_start=1.0, eps_end=0.01,
                                                   eps_decay=0.9995)
scores = np.array(scores)
losslist = np.array(avg_loss_list)
lengthlist = np.array(episode_length_list)
df = pd.DataFrame(scores, columns=['Scores'])
df['Loss'] = losslist
df['Episode length'] = lengthlist
print('\n', df.head())
df.to_csv("scores_Noisy.csv")
print("saved the scores")
print(30 * '#' + ' Training the agent Finished ' + 30 * '#')

"""
#SOURCES
#https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb
#https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/05.noisy_net.ipynb
#https://github.com/qfettes/DeepRL-Tutorials/blob/master/05.DQN-NoisyNets.ipynb"""
