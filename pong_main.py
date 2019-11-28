"""
This is an example on how to use the two player Wimblepong environment
with two SimpleAIs playing against each other
"""
import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import numpy as np
import argparse
import wimblepong
from PIL import Image
from Agent import Agent
import cv2
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
# Number of episodes/games to play
episodes = 100000

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = wimblepong.SimpleAi(env, opponent_id)
player = Agent(env, 1, player_id) #Agent(env, player_id) #introduce here our agent

# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())
matches =0
points_1 = 0
points_2 = 0
point_history1 = []
point_history2 = []
timesteps_history = []
reward_history = []

for i in range(0,episodes):

    obs1, obs2 = env.reset()
    #match loop
    match_done = False
    timesteps_list = []
    reward_list = []
    points=0
    while not match_done:
        done = False
        #point loop
        timesteps = 0
        rew_store =0
        while not done:
            timesteps += 1
            action1 = player.get_action(obs1) # our are all the 1 because we are the action 1 
            action2 = opponent.get_action()
            # Step the environment and get the rewards and new observations
            (obs1, obs2), (rew1, rew2), done, info = env.step((action1, action2))


            player.store_outcome(obs1, action1, rew1) 
            observation = (obs1, obs2)
            rew_store += rew1
            if not args.headless:
                env.render()
            if done:
                if rew1 == 10:
                    points_1 += 1
                else:
                    points_2 += 1
                observation= env.reset()
        player.update()
        timesteps_list.append(timesteps)
        reward_list.append(rew_store)
        #player.update_target_network()

        if points_1 >= 21 or points_2 >=21:
            match_done = True
        points +=1
    #player.update_target_network()

    print("episode " , i ," over. result = Bro : ",points_1, " AI : ",points_2, " avg_timesteps : ", np.array(timesteps_list).mean(), " avg_rew :", np.array(reward_list).mean())
    point_history1.append(int(points_1))
    point_history2.append(int(points_2))
    timesteps_history.append(np.array(timesteps_list).mean())
    reward_history.append(np.array(reward_list).mean())
    
    points_1 =0
    points_2 =0 
                 
    if i %100 ==0:
        torch.save(player.policy.state_dict(),"weights_%s_%d.mdl" % ("bro", i)) 
        # saving point made by bro
        np.savetxt('point_history_1.txt', np.array(point_history1), delimiter=',')
        np.savetxt('point_history_2.txt', np.array(point_history2), delimiter=',')
        np.savetxt('timestep_history.txt', np.array(timesteps_history), delimiter=',')
        np.savetxt('reward_history.txt', np.array(reward_history), delimiter=',')
    #if i%10==0:
    #    player.update_target_network()
