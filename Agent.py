from wimblepong import Wimblepong
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import cv2
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt


class PolicyConv(torch.nn.Module):
    def __init__(self, action_space, hidden=64):
        super().__init__()
        self.action_space = action_space
        self.hidden = hidden
        self.conv1 = torch.nn.Conv2d(1, 32, 3,stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        #self.conv2 = torch.nn.Conv2d(32, 64, 9, 2)
        #self.conv3 = torch.nn.Conv2d(64, 128, 9, 2)
        

        self.reshaped_size = 1568#128*11*11

        self.lstm = torch.nn.LSTM(self.reshaped_size, 256,num_layers=2)

        self.fc1_actor = torch.nn.Linear(256, self.hidden)
        self.fc1_critic = torch.nn.Linear(256, self.hidden)
        
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        
        self.fc2_value = torch.nn.Linear(self.hidden, 1)
        self.initialize()

    def initialize(self):
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        torch.nn.init.xavier_uniform_(self.fc1_actor.weight)
        torch.nn.init.xavier_uniform_(self.fc1_critic.weight)
        torch.nn.init.xavier_uniform_(self.fc2_mean.weight)
        torch.nn.init.xavier_uniform_(self.fc2_value.weight)
       

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)
        
        x = self.conv4(x)
        x = F.relu(x)

        x = F.relu(x)

        x = x.reshape(-1, 1,self.reshaped_size )
       
        x, _ = self.lstm(x)
        x_ac = self.fc1_actor(x)
        x_ac = F.relu(x_ac)
        

        x_mean = self.fc2_mean(x_ac)

        x_probs = F.softmax(x_mean, dim=-1)
        #print(x_probs)
        
        dist = Categorical(x_probs)

        x_cr = self.fc1_critic(x)
        x_cr = F.relu(x_cr)
    
        value = self.fc2_value(x_cr)

        return dist, value



class Agent(object):
    def __init__(self, env, num_inputs, player_id=1):
    
        if type(env) is not Wimblepong:
            raise TypeError("I'm not a very smart AI. All I can play is Wimblepong.")
        self.env = env
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        self.policy = PolicyConv(3, 400).to(self.device)
        self.policy.to(self.device)
        #second targt net 
        #self.target_net = PolicyConv(3, 200).to(self.device)
        #self.target_net.load_state_dict(self.policy.state_dict())
        
        self.prev_obs_t_1 = None
        self.prev_obs_t_2 = None

        #self.optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
        self.player_id = player_id  
        self.name = "Bro"
       
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        self.gamma = 0.99
        
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.values = [] # values saved during the training 

    def update(self):
        
        self.optimizer.zero_grad()
        action_probs = torch.stack(self.action_probs, dim=0).to(self.device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.device).squeeze(-1)
        values = torch.stack(self.values, dim=0).to(self.device).squeeze(-1) # values from the network
        self.states, self.action_probs, self.rewards, self.values = [], [], [], []

        discounted_rewards = self.discount_rewards(rewards,self.gamma)
        
        advantage =  discounted_rewards - values 
        advantage -= torch.mean(advantage)
        std = torch.std(advantage.detach()) if torch.std(advantage.detach()) >0 else 1
        advantage /=std

        weighted_probs = -action_probs*   advantage.detach()  
        

        actor_loss = weighted_probs.mean()
        critic_loss = advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss 
        
        ac_loss.backward()
        
        self.optimizer.step()

    def get_action(self, observation):
        x = self.preprocess(observation).to(self.device)
        dist, value = self.policy.forward(x)
        action = dist.sample()
        #print(action)
        self.values.append(value)
        self.action_probs.append( dist.log_prob(action))
        return action

    def get_name(self):
        return self.name

    def store_outcome(self, observation, action_taken, reward):
        self.states.append(observation)
        self.rewards.append(torch.Tensor([reward]))

    def reset(self):
        self.prev_obs_t_1 = None
        self.prev_obs_t_2 = None
    
    def preprocess(self, observation):

        #resized = scipy.misc.imresize(y, [80,80])
     

        observation = np.array(observation)
        observation = observation[::2, ::2].mean(axis=-1)
        # low pass and high pass for the image
        observation[observation <50 ] = 0.0
        observation[observation >50 ] = 255.0
        
        if self.prev_obs_t_1 is None:
            self.prev_obs_t_1 = np.zeros([100,100])
            self.prev_obs_t_2 = np.zeros([100,100])

        stack_ob =  observation + self.prev_obs_t_2 + self.prev_obs_t_1  #np.concatenate((self.prev_obs_t_1,self.prev_obs_t_2, observation), axis=-1) #  observation - self.prev_obs_t_2#
        
        stack_ob = torch.from_numpy(stack_ob).float().unsqueeze(0)
        stack_ob = stack_ob.reshape(1,100 ,100,1)
        stack_ob = stack_ob.transpose(1, 3)
        #plt.imshow(stack_ob, cmap='gray')
        #plt.show()
        self.prev_obs_t_1 = self.prev_obs_t_2 
        # removing the bar 
        observation[:,0:8] = 0.0
        observation[:,90:99] = 0.0 
        self.prev_obs_t_2 = observation* 0.8
         

        return stack_ob #torch.from_numpy(stack_ob).float().unsqueeze(0) #torch.from_numpy(x).float().unsqueeze(0)
    
    def load_model(self):
        weights = torch.load("weights_bro_7700.mdl",map_location=torch.device('cpu'))
        self.policy.load_state_dict(weights, strict=False)

    def discount_rewards(self,r, gamma):
        discounted_r = torch.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size(-1))):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def update_target_network(self):
        self.policy.load_state_dict(self.policy.state_dict())


