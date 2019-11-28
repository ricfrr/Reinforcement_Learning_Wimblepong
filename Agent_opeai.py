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
        #self.conv1 = torch.nn.Conv2d(1, 32, 3,stride=2, padding=1)
        #self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        #self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        #self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.reshaped_size = 400#1568#128*11*11
        self.lin_input = torch.nn.Linear(80*80, self.reshaped_size)


        self.fc1_actor = torch.nn.Linear(self.reshaped_size, self.hidden)
        #self.fc1_critic = torch.nn.Linear(self.reshaped_size, self.hidden)
        
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        
        #self.fc2_value = torch.nn.Linear(self.hidden, 1)
        self.initialize()

    def initialize(self):
        """ torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.conv4.weight) """
        torch.nn.init.xavier_uniform_(self.lin_input.weight)
        
        torch.nn.init.xavier_uniform_(self.fc1_actor.weight)
        #torch.nn.init.xavier_uniform_(self.fc1_critic.weight)
        torch.nn.init.xavier_uniform_(self.fc2_mean.weight)
        #torch.nn.init.xavier_uniform_(self.fc2_value.weight)
       

    def forward(self, x):
        """         
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
        """

        x= self.lin_input(x)
        x = F.relu(x)
       
        x_ac = self.fc1_actor(x)
        x_ac = F.relu(x_ac)
        
        x_mean = self.fc2_mean(x_ac)

        x_probs = F.softmax(x_mean, dim=-1)
        
        dist = Categorical(x_probs)

        #x_cr = self.fc1_critic(x)
        #x_cr = F.relu(x_cr)
    
        #value = self.fc2_value(x_cr)

        return dist #, value



class Agent(object):
    def __init__(self, env, num_inputs, player_id=1):
    
        #if type(env) is not Wimblepong:
        #    raise TypeError("I'm not a very smart AI. All I can play is Wimblepong.")
        self.env = env
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        self.policy = PolicyConv(2, 200).to(self.device)
        self.policy.to(self.device)
        #second targt net 
        #self.target_net = PolicyConv(3, 400).to(self.device)
        #self.target_net.load_state_dict(self.policy.state_dict())
        #self.target_net.to(self.device)
        
        self.prev_obs_t_1 = None
        self.prev_obs_t_2 = None

        #self.optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
        self.player_id = player_id  
        self.name = "Bro"
       
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=1e-4)
        self.gamma = 0.99
        
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.batch_size = 10
        self.ep_number  = 0

    def update(self):
        
        
        action_probs = torch.stack(self.action_probs, dim=0).to(self.device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.device).squeeze(-1)
        self.states, self.action_probs, self.rewards= [], [], []
        
        discounted_rewards = self.discount_rewards(rewards,self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards) 
        discounted_rewards /= torch.std(discounted_rewards)
        
        weighted_probs = -action_probs*discounted_rewards  
        
        loss = torch.mean(weighted_probs)
        loss.backward()

        self.ep_number +=1
        if self.ep_number % self.batch_size == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def get_action(self, observation):
        x = self.preprocess(observation).to(self.device)
        dist = self.policy.forward(x)
        action = dist.sample()+2
        #self.values.append(value)
        self.action_probs.append( dist.log_prob(action-2))
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
     
        observation = observation[35:195] # crop
        
        observation = observation[::2,::2,0] # downsample by factor of 2
        observation[observation == 144] = 0 # erase background (background type 1)
        observation[observation == 109] = 0 # erase background (background type 2)
        observation[observation != 0] = 1
        
        observation = np.array(observation).ravel()
        #observation = cv2.resize(observation, (int(100), int(100)))
        #observation = observation[::2, ::2].mean(axis=-1)
        # low pass and high pass for the image
        #observation[observation <50 ] = 0.0
        #observation[observation >50 ] = 255.0
        #observation[observation == 144] = 0 # erase background (background type 1)
        #observation[observation == 109] = 0 # erase background (background type 2)
        #observation[observation != 0] = 1 

        if self.prev_obs_t_1 is None:
            self.prev_obs_t_1 = np.zeros([80,80])
            self.prev_obs_t_2 = np.zeros(80*80)

        stack_ob =  observation - self.prev_obs_t_2 #+ self.prev_obs_t_1  #np.concatenate((self.prev_obs_t_1,self.prev_obs_t_2, observation), axis=-1) #  observation - self.prev_obs_t_2#
        #plt.imshow(stack_ob, cmap='gray')
        #plt.show()
        
        
        stack_ob = torch.from_numpy(stack_ob).float().unsqueeze(0)
        #stack_ob = stack_ob.reshape(1,80 ,80,1)
        #stack_ob = stack_ob.transpose(1, 3)
        #self.prev_obs_t_1 = self.prev_obs_t_2 
        # removing the bar 
        #observation[:,0:8] = 0.0
        #observation[:,90:99] = 0.0 
        self.prev_obs_t_2 = observation
         
        return stack_ob #torch.from_numpy(stack_ob).float().unsqueeze(0) #torch.from_numpy(x).float().unsqueeze(0)
    
    def load_model(self):
        weights = torch.load("weights_bro_9800.mdl",map_location=torch.device('cpu'))
        self.policy.load_state_dict(weights, strict=False)

    def discount_rewards(self,r, gamma):
        discounted_r = torch.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size(-1))):
            if r[t] != 0: running_add = 0 # because we are in openai 
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def update_target_network(self):
        self.policy.load_state_dict(self.policy.state_dict())


