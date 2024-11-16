import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

from _1_create_tetris_parts import merge_cubes,convert_parts_2d
from _2_verify_second_shape import inspect_parts
from _5_create_meshes import create_meshes

# TODO use cuda instead of cpu

# Values from 0 to 1
fun_meter = 1.0 # the more varaity of shapes the more fun it is (many same shapes is boring)
_2d_difficulty_meter = 1.0 # the harder to assembly the shape 8x8 the harder it is
_3d_difficulty_meter = 1.0 # the harder to assembly the shape 4x4x4 the harder it is

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(144, 144),
            nn.ReLU(),
            nn.Linear(144, 144),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

# Define the Value Network
class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(144, 144),
            nn.ReLU(),
            nn.Linear(144, 1)
        )

    def forward(self, x):
        return self.fc(x)

# PPO Hyperparameters
lr = 3e-4
gamma = 0.99
eps_clip = 0.2
epochs = 10

policy = PolicyNetwork()
value = ValueNetwork()
policy_optimizer = optim.Adam(policy.parameters(), lr=lr)
value_optimizer = optim.Adam(value.parameters(), lr=lr)

def generate_maze():
    """Generates a random maze as a flat tensor."""
    return torch.randint(0, 2, (144,), dtype=torch.float)

def ppo_update(states, actions, rewards):
    # Compute advantages
    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.tensor(rewards, dtype=torch.float)
    
    # Compute old action probabilities
    old_probs = policy(states).gather(1, actions.unsqueeze(1)).squeeze()

    # Calculate returns and advantages
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float)
    advantages = returns - value(states).squeeze()

    for _ in range(epochs):
        # Compute new probabilities and ratio
        probs = policy(states).gather(1, actions.unsqueeze(1)).squeeze()
        ratio = probs / (old_probs + 1e-10)

        # Compute loss using the PPO clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Optimize the policy
        policy_optimizer.zero_grad()
        print("aaaaaaaa")
        print(list(policy.parameters())[0].grad)
        policy_loss.backward()
        print(list(policy.parameters())[0].grad)
        policy_optimizer.step()

        # Value loss
        value_loss = nn.MSELoss()(value(states).squeeze(), returns.squeeze())
        value_optimizer.zero_grad()
        print("bbbbbbb")
        print(list(value.parameters())[0].grad)
        value_loss.backward()
        print(list(value.parameters())[0].grad)
        value_optimizer.step()

# Training Loop
tetris_parts = None
for episode in range(1000):
    result_connections = state = generate_maze()
    action_probs = policy(state)
    action_dist = Categorical(action_probs)
    action = action_dist.sample()

    reward = 0

    tetris_parts = merge_cubes(result_connections.view(3, 3, 4, 4) > 0.5)
    num_of_parts = len(tetris_parts)
    tetris_parts = convert_parts_2d(tetris_parts)

    # if int then not all parts 2d
    if isinstance(tetris_parts, int):
        reward = tetris_parts / num_of_parts
    else:
        print("================")
        print(tetris_parts)
        reward += 1 # good you made it here
        sorted_parts = inspect_parts(tetris_parts)
        for part in sorted_parts:
            # less repeast is more fun (total number of cubes is 64)
            # bigger part size is more fun (max size part 4x4=16)
            reward += part["size"]/part["repeats"]
    print(reward)

    # Collect data for PPO update
    ppo_update([state], [action], [reward])

if tetris_parts:
    create_meshes(tetris_parts)
    print("finished meshing all parts")
else:
    print("the parts are not good enough to mesh them")