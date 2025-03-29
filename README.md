# Path Planning for Differential Drive Robot Using Reinforcement Learning Dev Repo
This is the repository for development purpose.

## *update 1*
The robot can find a path to the goal for a specific one map trough a gap.
- Controlled Variable                     : Angular and linear velocity
- Reward Distribution                      : 1.5*normalized_lidar_distance_reading + 3.5*(1-normalized_angle_deviation) + 5*(normalized_change_in_distance_from_robot_to_goal)
- Collide Penalty with border or obstacle : -5 
