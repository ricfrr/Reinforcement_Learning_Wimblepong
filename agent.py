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
from collections import namedtuple



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

# define the replay memory class
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# define the network 
class DQN(torch.nn.Module):
    def __init__(self, action_space, hidden=64):
        super().__init__()
        self.hidden=hidden
        self.action_space = action_space
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
          
        self.reshaped_size = 10368
        # convolutional part of the network the network receive as a input 3 images
        self.conv1 = torch.nn.Conv2d(3, 16, 8, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 2)
        
        self.fc1 = torch.nn.Linear(self.reshaped_size, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, self.action_space)   
        
        self.initialize()

    # initialize the weight using the xavier initialization
    def initialize(self):
        
        torch.nn.init.xavier_uniform_(self.conv1.weight)  
        torch.nn.init.xavier_uniform_(self.conv2.weight)  
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, x):

        x= self.conv1(x)
        x = F.relu(x)

        x= self.conv2(x)
        x = F.relu(x)

        x = x.reshape(-1, self.reshaped_size)

        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        return x




class Agent(object):
    def __init__(self ):
    
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

        self.action_space=3
        self.ep_number  = 0

        self.policy_net = DQN(3, 200).to(self.device)
        self.target_net = DQN(3, 200).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # previous observation
        self.prev_obs_t1 = np.zeros((80,80,1))
        self.prev_obs_t2 = np.zeros((80,80,1))

        self.name = "Gigino"
       
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        # hyperparameters
        self.gamma = 0.99
        self.a = 3000 # a for GLIE decay
        self.epsilon = 1 
        self.decay = 0.99996
        self.batch_size = 128
        self.replay_buffer_size=70000
        
        # initialize the memory
        self.memory = ReplayMemory(self.replay_buffer_size)
    
    def update_network(self, updates=1):
        for _ in range(updates):
            self._do_network_update()

    # compute the network update sampling from the memory
    def _do_network_update(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        
        batch = Transition(*zip(*transitions))

        non_final_mask = 1-torch.tensor(batch.done, dtype=torch.uint8)
        non_final_next_states = [s for nonfinal,s in zip(non_final_mask,
                                     batch.next_state) if nonfinal > 0]
        next_state = torch.stack(batch.next_state).to(self.device)
        non_final_next_states = torch.stack(non_final_next_states).to(self.device) # TODO here the states has to be already stacked
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

    
        state_action_values = self.policy_net(state_batch.reshape(-1,3,80,80)).gather(1, action_batch.to(self.device))
        

        next_state_values = torch.zeros(self.batch_size).to(self.device)
        next_state_values[non_final_mask.bool()] = self.target_net(non_final_next_states.reshape(-1,3,80,80)).max(1)[0].detach()
        expected_state_action_values = reward_batch + (self.gamma *  next_state_values) 
        
        loss = F.smooth_l1_loss(state_action_values.squeeze(),
                                expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)
        self.optimizer.step()

            

    def get_action(self, state,epsilon=0.02,evaluate=True):

        self.ep_number += 1 
        self.epsilon = self.a/(self.a+self.ep_number) # GLIE UPDATE
        
        if evaluate:
            # if evaluation perform the previous observation update here
            state, st = self.preprocess_couple(np.array(state),  self.prev_obs_t2,self.prev_obs_t1)
            state.to(self.device) 
            self.prev_obs_t1 = self.prev_obs_t2
            self.prev_obs_t2 = st
        else:
            state  = self.preprocess(state).to(self.device)
            
        eps = max(epsilon,self.epsilon )
        sample = random.random()
        if sample > eps or evaluate:
            with torch.no_grad():
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item() 
        else:
            return random.randrange(self.action_space)

    def get_name(self):
        return self.name

    # copy the values of the policy net in the target net 
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # reset the previous two frames to a matrix of zero
    def reset(self):
        self.prev_obs_t1 = np.zeros((80,80,1))
        self.prev_obs_t2 = np.zeros((80,80,1))
        

    # store the transition in the replay buffer
    def store_transition(self, state, action, next_state, reward, done):
        
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32) 
        state, st_op = self.preprocess_couple(np.array(state), self.prev_obs_t2,self.prev_obs_t1)
        state.to(self.device)
        next_state, _ = self.preprocess_couple(np.array(next_state),st_op,self.prev_obs_t2)
        next_state.to(self.device)

        # update the previous frames 
        self.prev_obs_t1 = self.prev_obs_t2
        self.prev_obs_t2 = st_op

        # push the transition to the memory
        self.memory.push(state, action, next_state, reward, done)
    
    def preprocess(self, observation):
        # resize the image and perform the mean i order to obtain only one channel
        observation = np.array(observation)
        observation = cv2.resize(observation, (int(80), int(80))).mean(axis=-1)
        # convert the background in black and the players and the paddle in white
        observation[observation <50 ] = 0.0
        observation[observation >50 ] = 1
        observation = np.expand_dims(observation, axis=-1)
        observation = torch.Tensor(observation)
        
        # concatenate the observation with the next previous two observation
        stack_ob = np.concatenate((self.prev_obs_t1,self.prev_obs_t2, observation), axis=-1)
        stack_ob = torch.from_numpy(stack_ob).float().unsqueeze(0) 
        stack_ob = stack_ob.transpose(1, 3)

        return stack_ob

    # this function perform the same steps as the normal preprocess function but using as a parameter
    # all the observation, useful during evaluatian and saving in memory
    def preprocess_couple(self, observation, prev_obs_t2, prev_obs_t1 ):

        # resize the image and perform the mean i order to obtain only one channel
        observation = np.array(observation)
        observation = cv2.resize(observation, (int(80), int(80))).mean(axis=-1)
        # convert the background in black and the players and the paddle in white
        observation[observation <50 ] = 0.0
        observation[observation >50 ] = 1
        observation = np.expand_dims(observation, axis=-1)
        observation = torch.Tensor(observation)
       

        # concatenate the observation with the next previous two observation
        stack_ob = np.concatenate((prev_obs_t1,prev_obs_t2, observation), axis=-1)
        stack_ob = torch.from_numpy(stack_ob).float().unsqueeze(0) 
        stack_ob = stack_ob.transpose(1, 3)

        return stack_ob, observation

    # load the model 
    def load_model(self):
        weights = torch.load("model.mdl",map_location=torch.device(self.device)) # 340 top now # 480 vicino al 70% 1020 paura 77% # 1120 close to 70% # 1160 sopra 70% costante circa # 691/1000 -> 1040
        self.policy_net.load_state_dict(weights, strict=False)
        self.target_net.load_state_dict(weights, strict=False)
        self.target_net.eval()
       
       



