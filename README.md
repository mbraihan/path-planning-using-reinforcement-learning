# Path planning using Reinforcement Learning on different path

This repository contains code for path planning using Value Iteration, Policy Iteration and Q learning.

Two MDP problems I have chosen are related to Robotics, where the task of the robot is to
avoid the obstacle and the walls and reach the goal state from the start state. 

There can be many examples of the robot moving in the grid world. For example, idea of
mobile robots serving in a restaurants, or roaming around in a store. But in all of these cases the
environment is dynamic and it keeps changing with time. For purpose of this assignment I
have taken grid world as the occupancy grid map with static obstacles and the robot should
move from the start location to goal location avoiding these random obstacles.  

Map 1 has 144 states, while map 2 has 625 states. 

Start location of the robot is shown with blue state and goal location is shown with green state.

# This is visualization of the path traversing with random obstacles
![alt text](https://github.com/mraihan19/path_planning_using_reinforcement_learning/blob/main/traversing.gif)
