from environment import environment
from NAF_architecture import NAF_DQNN, noisy_policy
import torch


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# my_environment = environment(lane_num=5, screen_width=500, screen_height=1000,
#                              arena_width=2, robot_width=0.15, robot_length=0.3, num_lidarRay= 120, 
#                              num_lidarBatch=4, min_dBarrier=0.3*1.75, max_dBarrier=0.3*2.5, max_speed=1,max_time=35, dt=0.01, step_num=3)

# my_model = NAF_DQNN(hidden_size=256, action_size=2, state_size=7, lidar_size=120, 
#                  lidar_batch_num=4, max_action=[1, 1], device=device)


# policy = noisy_policy

# my_environment.test_arena()

# my_model.load_state_dict(torch.load("final_model.pth"))
# my_model.to(device)

# total_reward = 0
# state,lidar = my_environment.reset_env()
# done = False

# while not done:
#     state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
#     lidar_tensor = torch.tensor(lidar, dtype=torch.float32).unsqueeze(0).to(device)
#     action = policy((state_tensor, lidar_tensor), my_model, 0.001,device)
#     next_state, next_lidar, reward, done = my_environment.step(action[0],action[1])

#     total_reward += reward

#     print(f"\rtotal reward: {total_reward}; time passed: {my_environment.time_passed}; timeout: {my_environment.time_out}; done: {my_environment.episode_end} ", end="", flush=True)

    
#     state = next_state
#     lidar = next_lidar

# print("start replay")
# my_environment.run_replay()


barrier_pos = [[2,3],[3,1.5]]
barrier_radius = [0.5, 0.5]
target_pos = [4, 4]
robot_pos = [1, 1]

my_environment = environment(arena_width=5, arena_length=5, robot_width=0.15, robot_length=0.3, robot_pos=robot_pos, 
                             barrier_pos=barrier_pos, barrier_radius=barrier_radius, target_pos=target_pos, num_lidarRay=60,
                             num_lidarBatch=4, lidar_range= 0.45, max_speed=10, max_time=60, dt=0.01, step_num=3)

my_environment.reset_env()
my_environment.test_arena()
my_environment.user_control()
