# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:34:18 2024

@author: ayhan
"""

import pygame
import sys
from pacman_env import PacmanEnv
import time
import torch
import torch.optim as optim
from brain import ActorCritic

# Initialize Pygame
pygame.init()

#grid
GRID_SIZE = (5, 5)  # Number of rows and columns
CELL_SIZE = 40  # Size of each grid cell (pixels)




# Set up the game window
screen_width, screen_height = GRID_SIZE[0] * CELL_SIZE, GRID_SIZE[1] * CELL_SIZE
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Grid-Based Pac-Man")



# Initialize Pac-Man's position (grid coordinates)
pacman_x, pacman_y = 2,2

pacman_speed = 1  # Adjust the speed as needed

# # Pellet positions (you can add more)
#pellets = [(1, 2), (4, 1), (5, 7)]
pellets = [(1, 2), (3, 1)]

# Initialize the score
score = 0

env=PacmanEnv(GRID_SIZE,pellets,pacman_x,pacman_y)
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
reward=0
done=False
next_obs=env._get_observation()
ai_Play=False

clock=pygame.time.Clock()

ispretrained=True
if ispretrained:
    model = ActorCritic(input_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.00015)
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#model=None
if model!=None:
    next_obs=env._get_observation()
    state_tensor = torch.tensor(next_obs, dtype=torch.float32)
    policy, _ = model(state_tensor)
    print(policy)
    m_action = torch.multinomial(policy, 1).item()
    str_Action="none"
    if m_action==0:
        str_Action="left"
    elif m_action==1:
        str_Action="right"
    elif m_action==2:
      str_Action="up"
    elif m_action==3:
        str_Action="down"
    print("mdoel suggest action",str_Action)



# Main game loop
running = True
step_count=0
game_won=0
episode_count=0
ai_Play=True
while running:
    
    if ai_Play==False:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    #pacman_x -= pacman_speed
    
                    next_obs, reward, done, _ = env.step(0)
                    if model !=None:
                        state_tensor = torch.tensor(next_obs, dtype=torch.float32)
                        policy, _ = model(state_tensor)
                        print(policy)
                        m_action = torch.multinomial(policy, 1).item()
                        str_Action="none"
                        if m_action==0:
                            str_Action="left"
                        elif m_action==1:
                            str_Action="right"
                        elif m_action==2:
                            str_Action="up"
                        elif m_action==3:
                            str_Action="down"
                        print("model suggest action",str_Action)
                elif event.key == pygame.K_RIGHT:
                    #pacman_x += pacman_speed
    
                    next_obs, reward, done, _ = env.step(1)
                    if model !=None:
                        state_tensor = torch.tensor(next_obs, dtype=torch.float32)
                        policy, _ = model(state_tensor)
                        print(policy)
                        m_action = torch.multinomial(policy, 1).item()
                        str_Action="none"
                        if m_action==0:
                            str_Action="left"
                        elif m_action==1:
                            str_Action="right"
                        elif m_action==2:
                            str_Action="up"
                        elif m_action==3:
                            str_Action="down"
                        print("mdoel suggest action",str_Action)
                    
                elif event.key == pygame.K_UP:
                    #pacman_y -= pacman_speed
    
                    next_obs, reward, done, _ = env.step(2)
                    if model !=None:
                        state_tensor = torch.tensor(next_obs, dtype=torch.float32)
                        policy, _ = model(state_tensor)
                        print(policy)
                        m_action = torch.multinomial(policy, 1).item()
                        str_Action="none"
                        if m_action==0:
                            str_Action="left"
                        elif m_action==1:
                            str_Action="right"
                        elif m_action==2:
                            str_Action="up"
                        elif m_action==3:
                            str_Action="down"
                        print("mdoel suggest action",str_Action)
                    
                elif event.key == pygame.K_DOWN:
                    #pacman_y += pacman_speed
    
                    next_obs, reward, done, _ = env.step(3)
                    if model !=None:
                        state_tensor = torch.tensor(next_obs, dtype=torch.float32)
                        policy, _ = model(state_tensor)
                        print(policy)
                        m_action = torch.multinomial(policy, 1).item()
                        str_Action="none"
                        if m_action==0:
                            str_Action="left"
                        elif m_action==1:
                            str_Action="right"
                        elif m_action==2:
                            str_Action="up"
                        elif m_action==3:
                            str_Action="down"
                        print("mdoel suggest action",str_Action)
    #if ai playing
    else:
        state_tensor = torch.tensor(next_obs, dtype=torch.float32)
        policy, _ = model(state_tensor)
        print(policy)
        m_action = torch.multinomial(policy, 1).item()
        next_obs, reward, done, _ = env.step(m_action)
        step_count+=1
        
        
    #pygame.time.wait(500)
    # Clear the screen
    screen.fill((0, 0, 0))
    
    grid_font=pygame.font.Font(None, 18)
    #draw game elements render from obs
    game_element_index=0 
    for game_element_index in range(len(next_obs)):
        
        #draw pacman
        if next_obs[game_element_index] ==0.3:
            pacman_pos_x = game_element_index//GRID_SIZE[0]*CELL_SIZE
            pacman_pos_y=(game_element_index-game_element_index//GRID_SIZE[0]*GRID_SIZE[0])*CELL_SIZE
            pygame.draw.circle(screen, (255, 255, 0), (pacman_pos_x, pacman_pos_y), 20)
        #draw pellets    
        elif next_obs[game_element_index] == 0.6:
            pellet_pos_x = game_element_index//GRID_SIZE[0]*CELL_SIZE
            pellet_pos_y=(game_element_index-game_element_index//GRID_SIZE[0]*GRID_SIZE[0])*CELL_SIZE
            pygame.draw.circle(screen, (255, 255, 255), (pellet_pos_x,pellet_pos_y), CELL_SIZE//4)
            
        #optional draw grid number for pellet
        else:
            empty_pos_x = game_element_index//GRID_SIZE[0]*CELL_SIZE
            empty_pos_y=(game_element_index-game_element_index//GRID_SIZE[0]*GRID_SIZE[0])*CELL_SIZE
            grid_number=int(game_element_index)
            grid_number_text=grid_font.render(f"{grid_number}",True, (0, 255, 255))
            screen.blit(grid_number_text, (empty_pos_x,empty_pos_y))
    
    if len(env.pellets)==0:
        game_won+=1
        episode_count+=1
        next_obs=env.reset()
        step_count=0
    elif step_count>15:
        episode_count+=1
        next_obs=env.reset()
        step_count=0
            
    

    # # Draw Pacman
    # pacman_pos_x, pacman_pos_y = pacman_x * CELL_SIZE, pacman_y * CELL_SIZE
    # pygame.draw.circle(screen, (255, 255, 0), (pacman_pos_x, pacman_pos_y), 20)

    # # Draw pellets
    # for pellet in pellets:
    #     pygame.draw.circle(screen, (255, 255, 255), (pellet[0]*CELL_SIZE,pellet[1]*CELL_SIZE), CELL_SIZE//4)

    #     # # Check if Pacman ate a pellet
    #     if pygame.Rect(pellet[0]*CELL_SIZE , pellet[1]*CELL_SIZE, 10, 10).colliderect((pacman_x*CELL_SIZE, pacman_y*CELL_SIZE, CELL_SIZE//4, CELL_SIZE//4)):
    #         score += 10
    #         pellets.remove(pellet)  # Remove the eaten pellet

    # Display the score
    # font = pygame.font.Font(None, 36)
    # score_text = font.render(f"Score: {reward}", True, (255, 255, 255))
    # screen.blit(score_text, (0, 0))
    #display game won:
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"ep:{episode_count},w:{game_won}", True, (255, 255, 255))
    screen.blit(score_text, (0, 0))
    
    
    #display the grid numbers
    # grid_font=pygame.font.Font(None, 18)
    # for grid_x in range(GRID_SIZE[0]):
    #     for grid_y in range(GRID_SIZE[1]):
    #         grid_number=int(grid_y*10+grid_x)
    #         grid_number_text=grid_font.render(f"{grid_number}",True, (0, 255, 255))
    #         screen.blit(grid_number_text, (grid_x*CELL_SIZE,grid_y*CELL_SIZE))                                 
    
    
    # Update the display
    pygame.display.flip()
    clock.tick(5)
# Clean up and exit
pygame.quit()
sys.exit()



