from NAF_architecture import NAF_DQNN, noisy_policy
from train_algorithm import NAF_Trainning
from environment import environment
import torch
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = NAF_DQNN(hidden_size=400, action_size=2, state_size=11, max_action=[10, 10], device=device)

barrier_pos = [[2,3],[3,1.5]]
barrier_radius = [0.5, 0.5]
target_pos = [4, 4]
robot_pos = [1, 1]

my_environment = environment(arena_width=5, arena_length=5, robot_width=0.15, robot_length=0.3, robot_pos=robot_pos, 
                             barrier_pos=barrier_pos, barrier_radius=barrier_radius, target_pos=target_pos, num_lidarRay=180,
                             num_lidarBatch=4, lidar_range= 0.45, max_speed=10, max_time=60, dt=0.01, step_num=3)
policy = noisy_policy

trainning_algo = NAF_Trainning(model=model, gamma=0.99, tau=0.005, buffer_size=1000000, batch_size=100, 
                               env=my_environment, device=device, epsilon_max=1.5, epsilon_min=0.01, 
                               policy=policy, num_episodes=1000)


trainning_algo.test_arena()

trainning_algo.train()


os.makedirs("Model", exist_ok=True)

# Save the final model and metrics
trainning_algo.save_model("Model/final_model.pth")
trainning_algo.save_barriers("Model/barrier.csv")

trainning_algo.plot_return()
trainning_algo.run_replay_last_episode()