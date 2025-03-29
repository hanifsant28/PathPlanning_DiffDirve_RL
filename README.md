# Path Planning for Differential Drive Robot Using Reinforcement Learning
This project is a personal project to develop a path planning system by utlizing Deep Reinforcement Learning algorithm. Currently the algorithm that is used is Normalized Advantage Function (NAF) and in the future a Deep Deterministic Policy Gradient (DDPG) will also be used to see which algorithm works better for this task. The program is still under development and will always be updated to refine the robustness of the system. All the necessary file are inside the **code** folder.

## *update 1*
The robot can find a path to the goal for a specific one map trough a gap.
- Controlled Variable                     : Angular and linear velocity
- Reward Distribution                      : 1.5*normalized_lidar_distance_reading + 3.5*(1-normalized_angle_deviation) + 5*(normalized_change_in_distance_from_robot_to_goal)
- Collide Penalty with border or obstacle : -5 
