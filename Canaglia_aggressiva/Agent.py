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



class DQN(torch.nn.Module):
    def __init__(self, action_space, hidden=64):
        super().__init__()
        self.action_space = action_space
        self.hidden = hidden  
            

        self.reshaped_size = 10368

        self.conv1 = torch.nn.Conv2d(3, 16, 8, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 2)
        
        self.fc1 = torch.nn.Linear(self.reshaped_size, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, 3)   
        
        self.initialize()


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
    
        #self.env = env
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(3, 200).to(self.device)
        self.target_net = DQN(3, 200).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # previous observation
        self.prev_obs_t1 = np.zeros((80,80,1))
        self.prev_obs_t2 = np.zeros((80,80,1))

        
        

        #self.player_id = player_id  
        self.name = "Gigino"
       
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.gamma = 0.99

        self.a = 3000 # a for GLIE decay
        self.epsilon = 1 
        self.decay = 0.99996
        self.batch_size = 128
        self.ep_number  = 0
        self.state_space_dim = 2
        self.replay_buffer_size=70000
        self.action_space=3
        

        self.memory = ReplayMemory(self.replay_buffer_size)
        self.memory_opponent = ReplayMemory(self.replay_buffer_size)
        
    
    def update_network(self, updates=1):
        for _ in range(updates):
            self._do_network_update()
            #self._do_network_update(True)
    
    def _do_network_update(self, opponent=False):
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
        #x = self.preprocess(observation).to(self.device)
        self.ep_number += 1 
        self.epsilon =0 # self.a/(self.a+self.ep_number) # GLIE UPDATE
        
        if evaluate:
            # if evaluation no exploration involved 
            epsilon = 0
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

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def reset(self):
        self.prev_obs_t1 = np.zeros((80,80,1))
        self.prev_obs_t2 = np.zeros((80,80,1))
        self.prev_obs_opponent_t1 = np.zeros((80,80,1))
        self.prev_obs_opponent_t2 = np.zeros((80,80,1))
        

    def store_transition(self, state, action, next_state, reward, done):
        if self.prev_obs_t1 is None:
            self.prev_obs_t1 = np.zeros((80,80,1))
            self.prev_obs_t2 = np.zeros((80,80,1)) # store also for the final one
        
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32) 
        state, st_op = self.preprocess_couple(np.array(state), self.prev_obs_t2,self.prev_obs_t1)
        state.to(self.device)
        next_state, _ = self.preprocess_couple(np.array(next_state),st_op,self.prev_obs_t2)
        next_state.to(self.device)

        self.prev_obs_t1 = self.prev_obs_t2
        self.prev_obs_t2 = st_op

        self.memory.push(state, action, next_state, reward, done)
    
    def preprocess(self, observation):
        
        observation = np.array(observation)
        observation = cv2.resize(observation, (int(80), int(80))).mean(axis=-1)
        # low pass and high pass for the image
        observation[observation <50 ] = 0.0
        observation[observation >50 ] = 1
        #plt.imshow(observation, cmap='gray')
        #plt.show()

        observation = np.expand_dims(observation, axis=-1)
        observation = torch.Tensor(observation)
        
        if self.prev_obs_t1 is None:
            self.prev_obs_t1 = np.zeros((80,80,1))
            self.prev_obs_t2 = np.zeros((80,80,1))


        stack_ob = np.concatenate((self.prev_obs_t1,self.prev_obs_t2, observation), axis=-1)
        stack_ob = torch.from_numpy(stack_ob).float().unsqueeze(0) 
        stack_ob = stack_ob.transpose(1, 3)

        
        return stack_ob
        
    def preprocess_couple(self, observation, prev_obs_t2, prev_obs_t1 ):
        
        observation = np.array(observation)
        observation = cv2.resize(observation, (int(80), int(80))).mean(axis=-1)
        # low pass and high pass for the image
        observation[observation <50 ] = 0.0
        observation[observation >50 ] = 1

        observation = np.expand_dims(observation, axis=-1)
        observation = torch.Tensor(observation)
       

        stack_ob = np.concatenate((prev_obs_t1,prev_obs_t2, observation), axis=-1)
        stack_ob = torch.from_numpy(stack_ob).float().unsqueeze(0) 
        stack_ob = stack_ob.transpose(1, 3)

        return stack_ob, observation
    
    def load_model(self):
        weights = torch.load("weights_bro_340_wimblepong_DQN_5.mdl",map_location=torch.device(self.device)) # 340 top now
        
        optimizer = torch.load("optimizer_bro_880_wimblepong_DQN_3.mdl",map_location=torch.device(self.device))
        self.policy_net.load_state_dict(weights, strict=False)
        self.target_net.load_state_dict(weights, strict=False)
        self.target_net.eval()
       
        self.optimizer.load_state_dict(optimizer)




