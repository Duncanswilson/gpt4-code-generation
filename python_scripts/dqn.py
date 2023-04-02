import sys
import math
import gym
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable

pi = Variable(torch.FloatTensor([math.pi]))

def normal(x, mu, sigma_sq):
    a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
    return a*b

class REINFORCE:
    def __init__(self, policy, optimizer):
        self.model = policy
        #self.model = self.model.cuda()
        self.optimizer = optimizer
        self.model.train()

    def select_action(self, state):
        mu, sigma_sq = self.model(Variable(state))
        sigma_sq = F.softplus(sigma_sq)

        eps = torch.randn(mu.size())
        # calculate the probability
        action = (mu + sigma_sq.sqrt()*Variable(eps)).data
        prob = normal(action, mu, sigma_sq)
        entropy = -0.5*((sigma_sq+2*pi.expand_as(sigma_sq)).log()+1)

        log_prob = prob.log()
        return action, log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i]))).sum() - (0.0001*entropies[i]).sum()
        loss = loss / len(rewards)
		
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()

# Set up the Atari game environment
env = gym.make('MountainCar-v0')

# Define the neural network architecture
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the Reinforce algorithm
policy = DQN(env.observation_space.shape[0], env.action_space.n)
optimizer = optim.Adam(policy.parameters(), lr=1e-4)
agent = REINFORCE(policy, optimizer)

# Train the neural network using Reinforce
for episode in range(1000):
    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action_probs = policy(torch.from_numpy(state[0]).float())
        action = torch.multinomial(action_probs, 1).item()
        next_state, reward, done, info = env.step(action)
        agent.reinforce(reward)
        episode_reward += reward
        state = next_state
    print("Episode: {}, Reward: {}".format(episode, episode_reward))

# Test the trained neural network
state = env.reset()
done = False
while not done:
    action_probs = policy(torch.from_numpy(state).float())
    action = torch.argmax(action_probs).item()
    next_state, reward, done, info = env.step(action)
    state = next_state
    env.render()
