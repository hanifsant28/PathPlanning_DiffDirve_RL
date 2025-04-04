from NAF_architecture import NAF_DQNN, noisy_policy
from train_algorithm import NAF_Trainning
from environment import environment
import torch
import os
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("versuch 4")

barrier_pos = [[1.5,1.5]]
barrier_radius = [0.5]
target_pos = [2.5, 2.5]
robot_pos = [0.75, 0.75]
max_speed = [0.3, np.deg2rad(90)]

my_environment = environment(arena_width=3, arena_length=3, robot_width=0.15, robot_length=0.3, robot_pos=robot_pos, 
                             barrier_pos=barrier_pos, barrier_radius=barrier_radius, target_pos=target_pos, num_lidarRay=120,
                             num_lidarBatch=4, lidar_range= 0.45, max_speed=max_speed, max_time=240, dt=0.01, step_num=3)

model = NAF_DQNN(hidden_size=400, action_size=2, state_size=12, max_action=max_speed, device=device)

policy = noisy_policy

trainning_algo = NAF_Trainning(model=model, gamma=0.95, tau=0.005, buffer_size=1000000, batch_size=200, 
                               env=my_environment, device=device, epsilon_max=1.05, epsilon_min=0.001, 
                               policy=policy, num_episodes=1000, explor_ratio=0.5)


trainning_algo.test_arena()

trainning_algo.train()


os.makedirs("Model", exist_ok=True)

# Save the final model and metrics
trainning_algo.save_model("Model/final_model4.pth")
trainning_algo.save_barriers("Model/arena4.csv")

trainning_algo.plot_return()
trainning_algo.run_replay_last_episode()