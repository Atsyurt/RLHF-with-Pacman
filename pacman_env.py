# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 13:20:21 2024

@author: ayhan
"""

import gym
from gym import spaces
import numpy as np
import random


class PacmanEnv(gym.Env):
    def __init__(self,grid_size,pellets,pacman_x, pacman_y):
        super().__init__()
        self.wall_collision_penalty=-5
        self.progress_reward=-1
        self.eat_pellet_reward=20
        self.win_reward=30

        # Define the grid size and pellet positions
        self.grid_size = grid_size
        self.pellets = pellets
        self.initial_pellets = pellets.copy()
        
        
        # Define the action space (left, right, up, down)
        self.action_space = spaces.Discrete(4)

        # Define the observation space (Pac-Man position and pellet positions)
        #self.observation_space = spaces.Discrete(self.grid_size[0] * self.grid_size[1])
        self.observation_space = spaces.Box(low=0, high=2,shape=(grid_size[0]**2,),dtype=np.int8)
        # Initialize Pac-Man's position
        self.pacman_x, self.pacman_y = pacman_x, pacman_y

        # Initialize the score
        self.score = 0
        
        #get observation initally
        self.initial_onservation=self._get_observation()
        self.collisions=0
        self.way_list=[]
        self.super_succes_count=0

    def step(self, action):
        # Validate the action (0: left, 1: right, 2: up, 3: down)
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Update Pac-Man's position based on the action
        if action == 0:  # Left
            self.way_list.append("left")
            if self.pacman_x > 0:
                self.pacman_x -= 1
                self.score +=self.progress_reward
                
            else:
                self.score += self.wall_collision_penalty
                self.collisions+=1
        elif action == 1:  # Right
            self.way_list.append("right")
            if self.pacman_x < self.grid_size[0]-1:
                self.pacman_x += 1
                self.score +=self.progress_reward
            else:
                self.score += self.wall_collision_penalty
                self.collisions+=1
        elif action == 2:  # Up
            self.way_list.append("up")
            if self.pacman_y > 0:
                self.pacman_y -= 1
                self.score +=self.progress_reward
            else:
                    self.score += self.wall_collision_penalty
                    self.collisions+=1
        elif action == 3:  # Down
            self.way_list.append("down")
            if self.pacman_y < self.grid_size[1]-1:
                self.pacman_y += 1
                self.score +=self.progress_reward
            else:
                self.score += self.wall_collision_penalty
                self.collisions+=1

        # Check if Pac-Man ate a pellet
        for pellet in self.pellets:
            if (self.pacman_x, self.pacman_y) == pellet:
                self.score += self.eat_pellet_reward
                self.pellets.remove(pellet)  # Remove the eaten pellet
                self.way_list.append("eat")
                break
            
        if len(self.pellets)==0:
            self.score +=self.win_reward
            print("you won bro congratulations")
            if self.collisions<=1:
                self.score +=5*self.win_reward
                print("you super uper won bro congratulations")
                self.super_succes_count+=1
                print(self.way_list)
                
            elif self.collisions <= 3 :
                self.score =3*self.win_reward
                print("  very good win ")
                self.super_succes_count+=0.5
            elif self.collisions <= 5 :
                    self.score =2*self.win_reward
                    print("  good win")
            elif self.collisions <= 8 :
                    self.score =1*self.win_reward
                    print(" normal win")
            else:
                    self.score =self.win_reward - self.collisions*10
                    print("  but yoou are not so smart bad dog")
            


        # Determine whether the episode is done (no more pellets)
        done = len(self.pellets) == 0

        # Return the next observation, reward, done flag, and additional info
        observation = self._get_observation()
        reward = self.score
        return observation, reward, done, {}
    
    def _get_observation(self):
        # Create a grid representation with Pac-Man and pellets
        grid = np.zeros((self.grid_size[0], self.grid_size[1]))
        grid[(self.pacman_x,self.pacman_y)] = 0.3
        # Pac-Man
        for pellet_pos in self.pellets:
            grid[pellet_pos] = 0.6  # Pellets
        #return grid.flatten()
        grid=np.reshape(grid, (self.grid_size[0]*self.grid_size[1]))
        #print(grid)
        return grid

    def reset(self):
        # Reset the environment to initial state
        self.pacman_x, self.pacman_y = 2, 2
        #self.pacman_x, self.pacman_y =random.randint(1, self.grid_size[0])-1,random.randint(1, self.grid_size[1])-1
        self.score = 0
        
        self.pellets = self.initial_pellets.copy()
        if random.randint(1,5) < 2:
            self.pellets[0]=(random.randint(1,5)-1,random.randint(1,5)-1)
            #self.pellets[0]=self.pellets[0].copy()
            print("lets change pellet super fun and easy peasy")
        self.collisions=0
        self.way_list=[]
        return self._get_observation()

    def render(self, mode="human"):
        # Optionally, implement a rendering function to visualize the game
        pass



# GRID_SIZE=(10, 10)
# pacman_x,pacman_y = 5, 5
# pellets = [(1, 2), (4, 1), (5, 7)]
# env=PacmanEnv(GRID_SIZE,pellets,pacman_x,pacman_y)
# ninital_obs=env.reset()
# print(env.observation_space.sample())
# action = env.action_space.sample()

# # Take a step in the environment
# next_obs, reward, done, _ = env.step(action)

# observation_shape = env.observation_space.shape
# print(f"Observation shape: {observation_shape}")

# grid = np.zeros((GRID_SIZE[0],GRID_SIZE[1]), dtype=np.int8)
# pacman_position=[4,5]
# grid[(pacman_position[0],pacman_position[1])] = 1  # Pac-Man
# for pellet_pos in pellets:
#     grid[pellet_pos] = 2  # Pellets


# next_obs=grid.flatten()