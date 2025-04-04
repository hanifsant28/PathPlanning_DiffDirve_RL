import numpy as np
import torch
from torch.optim import Adam
import random
import copy
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt


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
                 num_episodes,
                 explor_ratio):
        
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
        self.explore_ratio = explor_ratio

    def collect_experience(self):
        """
        Function to collect experiences.
        """
        self.total_reward = 0
        state = self.env.reset_env()
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.policy(state_tensor, self.model, self.epsilon,self.device, self.env.max_speed)
            next_state, reward, done = self.env.step(action[0],action[1])

            #store the result in bufffer
            self.buffer.append([state, action, reward, next_state, done])
            
            self.total_reward += reward
            print(f"\rY Pos: {self.env.y_real}; X Pos: {self.env.x_real}; total reward: {self.total_reward}; time passed: {self.env.time_passed} ", end="", flush=True)
            state = next_state


            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)

            if len(self.buffer) > self.batch_size:
                self.update_network()

    
    def update_network(self):
        """
        Function to update the neural network if the buffer already more than the batch size.
        """
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Compute the target value
        next_values = self.target_model.value((next_states)).squeeze()
        target = rewards + (1 - dones) * self.gamma * next_values

        # compute the current values
        values = self.model.forward(states,actions).squeeze()

        loss = torch.nn.functional.mse_loss(values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def polyak_update(self):
        """
        Function to do polyak update so that it is more stable.
        """
        for target_param, online_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1 - self.tau) * target_param.data)

    
    def update_epsilon(self, episode):
        """
        Function to update the value of epsilon for exploration policy.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon_max - (self.epsilon_max-self.epsilon_min)*(episode/(self.num_episodes*self.explore_ratio)))

    
    def train(self):
        """
        Main function for trainning the model.
        """
        self.return_each_episode = []
        for episode in tqdm(range(self.num_episodes)):
            self.collect_experience()
            self.update_epsilon(episode)
            self.polyak_update()
            self.return_each_episode.append(self.total_reward)


    def test_arena(self):
        """
        Function to show the arena used for trainning
        """
        self.env.test_arena()
    

    def run_replay_last_episode(self):
        """
        Running the last episode replay of the trainning
        """
        self.env.run_replay()


    def save_model(self, filepath):
        """
        Function to save the Model parameters.
        """
        torch.save(self.model.state_dict(), filepath)


    def save_barriers(self, filepath="Model/arena.csv"):
        """
        Function to save the barrier position to be used for testing.
        """
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for pos, radius in zip(self.env.barrier_pos,self.env.barrier_radius):
                writer.writerow([pos[0], pos[1], radius])
            
            writer.writerow(["-","-","-"])
            writer.writerow([self.env.start_x,self.env.start_y,
                             self.env.target_pos[0],self.env.target_pos[1],
                             self.env.arena_width, self.env.arena_length])


    def plot_return(self):
        """
        Function to plot the graphic of return each episode
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.return_each_episode, label='Return per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title('Return per Episode')
        plt.legend()
        plt.grid(True)
        plt.show()




def load_model( model,filepath,device):
    """
    Function to load the model

    Parameters:
        - filepath (string): Desired directory.
        - device  (string): Desired device for processing the model.
    
    Returns:
        - model : The loaded model
    """

    model.load_state_dict(torch.load(filepath))
    model.to(device)
    return model


def load_barriers(filepath="Model/arena.csv"):
    """
    Function to load the barrier position to be used for testing.

    Parameters:
        - filepath (string): Desired directory.

    Returns:
        - barrier_pos_pair (array of float): Array of the posiiton pair of the barrier.
    """
    pos = []
    radius = []
    start_pos = []
    arena_width = 0
    arena_length = 0
    target_pos = []
    done_read_barrier = False

    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == "-":
                done_read_barrier = True
            elif done_read_barrier == False:
                coor = [float(row[0]),float(row[1])]
                pos.append(coor)
                radius.append(float(row[2]))
            elif done_read_barrier:
                start_pos = [float(row[0]), float(row[1])]
                target_pos = [float(row[2]), float(row[3])]
                arena_width = float(row[4])
                arena_length = float(row[5])

    return [pos,radius,start_pos,target_pos,arena_width,arena_length]