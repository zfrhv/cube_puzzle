from _1_create_tetris_parts import merge_cubes
from _2_verify_second_shape import inspect_parts
from _5_create_meshes import create_meshes

import torch
import torch.nn as nn
import torch.optim as optim

# Values from 0 to 1
fun_meter = 1.0 # the more varaity of shapes the more fun it is (many same shapes is boring)
_2d_difficulty_meter = 1.0 # the harder to assembly the shape 8x8 the harder it is
_3d_difficulty_meter = 1.0 # the harder to assembly the shape 4x4x4 the harder it is

# Connect the cubes

class PiecesGenerator(nn.Module):
    def __init__(self):
        super(PiecesGenerator, self).__init__()
        self.fc1 = nn.Linear(3, 144)
        self.fc2 = nn.Linear(144, 144)
        self.fc3 = nn.Linear(144, 144)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # Sigmoid to get output between 0 and 1 for connections

# Initialize the model, optimizer, and latent space
pieces_generator = PiecesGenerator()
optimizer = optim.Adam(pieces_generator.parameters(), lr=0.001)

tetris_parts = None

last_feedback = 0


# Training loop
for iteration in range(1000):
    result_connections = pieces_generator(torch.tensor([fun_meter, _2d_difficulty_meter, _3d_difficulty_meter]))

    feedback = 1.0

    tetris_parts = merge_cubes(result_connections.view(3, 3, 4, 4) >= 0.5)

    # TODO add memory to roll back in case went bad

    # if int then not all parts successful
    if isinstance(tetris_parts, int):
        feedback = tetris_parts/64
        print(tetris_parts)
        print("0", feedback)
    else:
        sorted_parts = inspect_parts(tetris_parts)
        for part in sorted_parts:
            # less repeast is more fun (total number of cubes is 64)
            # bigger part size is more fun (max size part 4x4=16)
            feedback = feedback * (1 - part["repeats"]/64) * part["size"] / 16
        # give it the minimum
        feedback = feedback + 0.1 * (1 - feedback)
        print("1", feedback)

    # print(result_connections)

    # TODO need PPO approach

    # Calculate new random target
    new_random_target = torch.rand_like(result_connections)
    new_random_target = result_connections * feedback + new_random_target * (1 - feedback) # the worse the more significant move
    loss = nn.BCEWithLogitsLoss()(result_connections, new_random_target)

    # Backpropagation and optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # optimizer.zero_grad()
    # if feedback >= last_feedback:
    #     loss.backward()
    #     optimizer.step()
    #     # reset_optimizer_state(optimizer)
    # last_feedback = feedback

if tetris_parts:
    create_meshes(tetris_parts)
    print("finished meshing all parts")
else:
    print("the parts are not good enough to mesh them")








import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Define the Policy and Value Networks (Actor-Critic architecture)
class PPOModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_policy = nn.Linear(128, output_dim)  # Policy head (actor)
        self.fc_value = nn.Linear(128, 1)  # Value head (critic)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.fc_policy(x), dim=-1)
        value = self.fc_value(x)
        return policy, value

# PPO loss function with clipping
def ppo_loss(old_log_probs, new_log_probs, advantages, old_values, values, epsilon=0.2, gamma=0.99, lam=0.95):
    # Calculate the ratio of the new and old probabilities
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # Calculate the clipped surrogate objective
    surrogate1 = ratio * advantages
    surrogate2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    
    # Final surrogate loss
    policy_loss = -torch.min(surrogate1, surrogate2).mean()
    
    # Value function loss (MSE)
    value_loss = F.mse_loss(values, old_values + advantages)

    # Entropy loss (to encourage exploration)
    entropy_loss = -torch.mean(torch.sum(new_log_probs * torch.exp(new_log_probs), dim=-1))

    # Total loss
    total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
    return total_loss

# Define the PPO agent
class PPOAgent:
    def __init__(self, env, input_dim, output_dim, lr=3e-4, gamma=0.99, epsilon=0.2, lam=0.95, batch_size=64, epochs=10):
        self.env = env
        self.model = PPOModel(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.lam = lam
        self.batch_size = batch_size
        self.epochs = epochs

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        policy, _ = self.model(state)
        action_prob = policy.detach().numpy()
        action = np.random.choice(len(action_prob), p=action_prob)
        log_prob = torch.log(policy[action])
        return action, log_prob

    def update(self, states, actions, log_probs_old, rewards, dones, values_old):
        advantages = self.compute_advantages(rewards, dones, values_old)
        for _ in range(self.epochs):
            for i in range(0, len(states), self.batch_size):
                batch_states = torch.tensor(states[i:i+self.batch_size], dtype=torch.float32)
                batch_actions = torch.tensor(actions[i:i+self.batch_size], dtype=torch.long)
                batch_log_probs_old = torch.tensor(log_probs_old[i:i+self.batch_size], dtype=torch.float32)
                batch_rewards = torch.tensor(rewards[i:i+self.batch_size], dtype=torch.float32)
                batch_dones = torch.tensor(dones[i:i+self.batch_size], dtype=torch.float32)
                batch_values_old = torch.tensor(values_old[i:i+self.batch_size], dtype=torch.float32)

                policy, value = self.model(batch_states)
                new_log_probs = torch.log(policy.gather(1, batch_actions.view(-1, 1)))

                loss = ppo_loss(batch_log_probs_old, new_log_probs, advantages, batch_values_old, value)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def compute_advantages(self, rewards, dones, values_old):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_old[t+1] * (1 - dones[t]) - values_old[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages, dtype=torch.float32)

    def train(self, max_episodes):
        for episode in range(max_episodes):
            states, actions, log_probs_old, rewards, dones, values_old = [], [], [], [], [], []
            state = self.env.reset()

            done = False
            while not done:
                action, log_prob = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                log_probs_old.append(log_prob.item())
                rewards.append(reward)
                dones.append(done)
                values_old.append(self.model(torch.tensor(state, dtype=torch.float32))[1].item())

                state = next_state

            # Update the model after each episode
            self.update(states, actions, log_probs_old, rewards, dones, values_old)

# Example environment (replace with your maze generation environment)
class DummyEnv:
    def reset(self):
        return np.random.random(4)  # dummy state

    def step(self, action):
        next_state = np.random.random(4)  # dummy state
        reward = np.random.random()  # dummy reward
        done = np.random.random() > 0.95  # random done condition
        return next_state, reward, done, {}

# Example of running the agent
env = DummyEnv()
input_dim = 4  # example state dimension (replace with actual state space)
output_dim = 2  # example action space (replace with actual number of actions)
agent = PPOAgent(env, input_dim, output_dim)

agent.train(max_episodes=1000)
