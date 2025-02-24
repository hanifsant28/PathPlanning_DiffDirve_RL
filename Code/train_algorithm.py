import numpy as np
import torch
from torch.optim import Adam
import random
import copy
from tqdm import tqdm
import csv


class NAF_Trainning:
    def __init__(self, 
                 model, 
                 gamma, 
                 tau, 
                 buffer_size, 
                 batch_size, 
                 env, device, 
                 epsilon_max,
                 epsilon_min,
                 policy,
                 num_episodes):
        
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.env = env
        self.epsilon_max = epsilon_max
        self.epsilon = self.epsilon_max
        self.epsilon_min = epsilon_min
        self.device = device
        self.policy = policy
        self.num_episodes = num_episodes

        self.total_reward = 0
        self.return_each_episode = []

    def collect_experience(self):
        self.total_reward = 0
        state,lidar = self.env.reset_env()
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            lidar_tensor = torch.tensor(lidar, dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.policy((state_tensor, lidar_tensor), self.model, self.epsilon,self.device)
            next_state, next_lidar, reward, done = self.env.step(action[0],action[1])

            

            #store the result in bufffer
            self.buffer.append([state, lidar, action, reward, next_state, next_lidar, done])
            
            self.total_reward += reward
            print(f"\rtotal reward: {self.total_reward}; time passed: {self.env.time_passed} ", end="", flush=True)
            state = next_state
            lidar = next_lidar

            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)

            if len(self.buffer) > self.batch_size:
                self.update_network()

    def update_network(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, lidars, actions, rewards, next_states, next_lidars, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        lidars = torch.tensor(lidars, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        next_lidars = torch.tensor(next_lidars, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Compute the target value
        next_values = self.target_model.value((next_states, next_lidars)).squeeze()
        target = rewards + (1 - dones) * self.gamma * next_values

        # compute the current values
        values = self.model.forward((states, lidars),actions).squeeze()

        loss = torch.nn.functional.mse_loss(values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def polyak_update(self):
        for target_param, online_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1 - self.tau) * target_param.data)

    
    def update_epsilon(self, episode):
        self.epsilon = max(self.epsilon_min, self.epsilon_max * (1 - episode / (0.5* self.num_episodes)) + self.epsilon_min)

    
    def train(self):
        for episode in tqdm(range(self.num_episodes)):
            self.collect_experience()
            self.update_epsilon(episode)
            self.polyak_update()
            self.return_each_episode.append(self.total_reward)

        # # Save the final model and metrics
        # self.save_model("final_model.pth")
        # self.save_metrics("training_metrics.csv")


    def test_arena(self):
        self.env.test_arena()
    
    def run_replay_last_episode(self):
        self.env.run_replay()

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.model.to(self.device)

    def save_metrics(self, filepath):
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Episode", "Total Reward"])
            for episode, reward in enumerate(self.return_each_episode):
                writer.writerow([episode, reward])
            