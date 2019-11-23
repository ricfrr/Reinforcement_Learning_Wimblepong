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
from utils import to_gray_scale_and_downsample

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
win1 = 0
for i in range(0,episodes):
    if i%21==0:
        print("-------Match",matches)
        
        matches +=1
    done = False
    observation = env.reset()
    observation = to_gray_scale_and_downsample(observation[0])
    while not done:
        action1 = player.get_action(observation) # our are all the 1 because we are the action 1 
        action2 = opponent.get_action()
        # Step the environment and get the rewards and new observations
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
        #convert to grayscale
        ob1_g_d =  to_gray_scale_and_downsample(ob1)
        player.store_outcome(ob1_g_d, action1, rew1) 
        #img = Image.fromarray(ob1)
        #img.save("ob1.png")
        #img = Image.fromarray(ob2)
        #img.save("ob2.png")
        # Count the wins
        if i%21==0:
            win1 = 0
        
        if rew1 == 10:
            win1 += 1
        if not args.headless:
            env.render()
        if done:
            observation= env.reset()
            print("episode {} over. Broken WR: {:.3f}".format(i%21, win1/(i%21+1)))
    
    player.update()