from environment import environment
from NAF_architecture import NAF_DQNN, noisy_policy
import torch
import numpy as np
from train_algorithm import load_barriers, load_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_filepath = "Model/final_model2.pth"


max_speed = [0.3, np.deg2rad(90)]

model = NAF_DQNN(hidden_size=400, action_size=2, state_size=11, max_action=max_speed, device=device)
barrier_pos,barrier_radius,robot_pos,target_pos,arena_length,arena_width = load_barriers("Model/arena2.csv")
model = load_model(model=model,filepath=model_filepath,device=device)

# my_environment = environment(arena_width=arena_width, arena_length=arena_length, robot_width=0.15, robot_length=0.3, robot_pos=robot_pos, 
#                              barrier_pos=barrier_pos, barrier_radius=barrier_radius, target_pos=target_pos, num_lidarRay=120,
#                              num_lidarBatch=4, lidar_range= 0.45, max_speed=max_speed, max_time=120, dt=0.01, step_num=3)

# my_environment.test_arena()
# total_reward = 0

# state = my_environment.reset_env()
# done = False
# while not done:
#     state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
#     action = noisy_policy(state=state_tensor,net=model,epsilon=0.001,device=device,max_speed=max_speed)
#     next_state, reward, done = my_environment.step(action[0],action[1])

#     total_reward += reward

#     print(f"\rtotal reward: {total_reward}; time passed: {my_environment.time_passed}; timeout: {my_environment.time_out}; done: {my_environment.episode_end} ", end="", flush=True)

    
#     state = next_state

# print("start replay")
# my_environment.run_replay()


# barrier_pos = [[1.25,2.25],[2,1.25]]
# barrier_radius = [0.25, 0.25]
# target_pos = [2.5, 2.5]
# robot_pos = [0.75, 0.75]
max_speed = [0.3, np.deg2rad(100)]

my_environment = environment(arena_width=3, arena_length=3, robot_width=0.15, robot_length=0.3, robot_pos=robot_pos, 
                             barrier_pos=barrier_pos, barrier_radius=barrier_radius, target_pos=target_pos, num_lidarRay=180,
                             num_lidarBatch=10, lidar_range= 0.45, max_speed=max_speed, max_time=60, dt=0.01, step_num=3)

my_environment.reset_env()
my_environment.test_arena()
my_environment.user_control()
