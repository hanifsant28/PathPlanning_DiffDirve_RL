from NAF_architecture import NAF_DQNN, noisy_policy
from train_algorithm import NAF_Trainning
from environment import environment
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model = NAF_DQNN(hidden_size=256, action_size=2, state_size=7, lidar_size=120, 
                 lidar_batch_num=4, max_action=[1, 1], device=device)

my_environment = environment(lane_num=5, screen_width=500, screen_height=1000,
                             arena_width=2, robot_width=0.15, robot_length=0.3, num_lidarRay= 120, 
                             num_lidarBatch=4, min_dBarrier=0.3*1.75, max_dBarrier=0.3*2.5, max_speed=1,max_time=65, dt=0.01, step_num=3)
policy = noisy_policy

trainning_algo = NAF_Trainning(model=model, gamma=0.99, tau=0.005, buffer_size=100000, batch_size=64, 
                               env=my_environment, device=device, epsilon_max=0.9, epsilon_min=0.1, 
                               policy=policy, num_episodes=1500)



trainning_algo.train()

# Save the final model and metrics
trainning_algo.save_model("final_model.pth")
trainning_algo.save_metrics("training_metrics.csv")
