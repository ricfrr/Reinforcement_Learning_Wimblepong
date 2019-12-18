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
from agent import Agent
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
player = Agent() #Agent(env, player_id) #introduce here our agent
player.load_model()
# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())
# counter useful for saving data while training
matches =0
points_1 = 0
points_2 = 0
point_history1 = []
point_history2 = []
timesteps_history = []
reward_history = []
running_reward = None
timesteps = 0 

for i in range(0,episodes):

    obs1, obs2 = env.reset()
    observation1 = obs1
    observation2 = obs2
    #set to zero all the variable of interest
    match_done = False
    timesteps_list = []
    reward_list = []
    points=0
    rew_store = 0
    
    while not match_done:
        done = False
        #point loop
        while not done:
            timesteps += 1
            action1 = player.get_action(obs1) 
            action2 = opponent.get_action()
            # Step the environment and get the rewards and new observations
            (obs1, obs2), (rew1, rew2), done, info = env.step((action1, action2))
            player.store_transition(observation1,action1,obs1 ,rew1/10.0, done) 
            
            # update the network every 4 timesteps using the global counter
            if timesteps % 4 ==0:
                player.update_network()
            observation1 = obs1

            # clip the reward to +/- 1
            rew_store += rew1/10.0

            if not args.headless:
                env.render()
            if done:
                # compute the points done 
                if rew1 == 10:
                    points_1 += 1
                else:
                    points_2 += 1
                # reset player and environment at the end of the episode
                obs1,obs2= env.reset()
                player.reset() 
            # update the target network every 1000 timesteps
            if timesteps % 1000 == 0:
                player.update_target_network()
        reward_list.append(rew_store)

        if points_1 >= 21 or points_2 >=21:
            match_done = True
        
    # store the data from the math    
    timesteps_list.append(timesteps)
    point_history1.append(int(points_1))
    point_history2.append(int(points_2))
    timesteps_history.append(np.array(timesteps_list).mean())
    reward_history.append(np.array(reward_list).mean())
    
    # compute the running reward and print performance
    running_reward = rew_store if running_reward is None else running_reward * 0.99 + rew_store * 0.01
    print("episode " , i ," over. result = Gigino : ",points_1, " AI : ",points_2, " avg_timesteps : ", np.array(timesteps_list).mean(), " runnin reward :", running_reward)
    
    points_1 =0
    points_2 =0 
   
    # every 20 match update the data on disk and export the model and optimizer             
    if i %20 ==0:
        torch.save(player.policy_net.state_dict(),"weights_%s_%d_wimblepong_DQN_5.mdl" % ("g", i)) 
        torch.save(player.optimizer.state_dict(),"optimizer_%s_%d_wimblepong_DQN_5.mdl" % ("g", i))
        # saving point made by bro
        np.savetxt('point_history_1_DQN_5.txt', np.array(point_history1), delimiter=',')
        np.savetxt('point_history_2_DQN_5.txt', np.array(point_history2), delimiter=',')
        np.savetxt('timestep_history_DQN_5.txt', np.array(timesteps_history), delimiter=',')
        np.savetxt('reward_history_DQN_5.txt', np.array(reward_history), delimiter=',')
 
