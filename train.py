# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 18:51:58 2024

@author: ayhan
"""

import torch
from brain import ActorCritic
import torch.nn as nn
import torch.optim as optim
import gym
from pacman_env import PacmanEnv
import numpy as np
import random

GRID_SIZE = (5, 5)  # Number of rows and columns
CELL_SIZE = 40  # Size of each grid cell (pixels)

# Initialize Pac-Man's position (grid coordinates)
pacman_x, pacman_y = 2, 2

pacman_speed = 1  # Adjust the speed as needed

# # Pellet positions (you can add more)
pellets = [(1, 2), (3, 1)]

# Initialize the score


env=PacmanEnv(GRID_SIZE,pellets,pacman_x,pacman_y)
reward=0
done=False

next_obs=env._get_observation()
nextt_obs=np.reshape(next_obs, (5,5))
# next_obs=env.reset()
# next_obs=np.reshape(next_obs, (5,5),axis=1)

score = 0
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
model = ActorCritic(input_size, output_size)


optimizer = optim.Adam(model.parameters(), lr=0.00015)
step=0
total_loss=None

ispretrained=True
if ispretrained:
    checkpoint = torch.load('checkpoint24node.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])




for episode in range(60000):
    state = env.reset()
    done = False
    step=0

    while not done and step<=15:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        policy, value = model(state_tensor)
        #see
        #policy.detach().numpy()
        #value.detach().numpy().item()
        #print(policy)
        if random.randint(1,5) < 4:
            action = torch.multinomial(policy, 1).item()
        #print(policy)
        else:
            action = torch.argmax(policy).item()
        # copy_policy=policy.view(-1)
        # #print(policy)
        # probs=copy_policy.detach().numpy()
        # action=np.argmax(probs)
    
        next_state, reward, done, _ = env.step(action)

        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        _, next_value = model(next_state_tensor)

        advantage = reward + 0.99 * next_value - value

        # Actor loss
        actor_loss = -torch.log(policy[action]) * advantage
        entropy_loss =-torch.log(policy[action]).mean() 
        # Critic loss
        critic_loss = advantage.pow(2)

        total_loss = actor_loss + critic_loss+entropy_loss*0.01

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        state = next_state
        step=step+1

    print(f"Episode {episode+1}: Total reward = {reward}, Pellet number = {len(env.pellets)}, Step number = {step} collisions= {env.collisions} ssc={env.super_succes_count}")
    if env.super_succes_count>20000:
        print("i should stop the training kupo i was doing great.")
        print("",env.way_list)
        break
    # if episode/env.super_succes_count<=3:
    #     print("i should stop the training kupo i was doing great.")
    #     break
        
    # if env.collisions==0:
    #     if len(env.pellets)==0 and episode >50:
    #             print("i should stop the training kupo i was doing great.")
    #             print("",env.way_list)
    #             break




# Assuming 'model' is your PyTorch model
#torch.save(model, 'model.pth')
# torch.save(model.state_dict(), 'model_params.pth')
# model = ActorCritic(input_size, output_size)
# model.load_state_dict(torch.load('model_params.pth'))



torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'checkpoint24node.pth')