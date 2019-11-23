from wimblepong import Wimblepong
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyConv(torch.nn.Module):
    def __init__(self, action_space, hidden=64):
        super().__init__()
        self.action_space = action_space
        self.hidden = hidden
        self.conv1 = torch.nn.Conv2d(2, 32, 3, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 2)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, 2)
        self.reshaped_size = 128*11*11
        self.fc1_actor = torch.nn.Linear(self.reshaped_size, self.hidden)
        self.fc1_critic = torch.nn.Linear(self.reshaped_size, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        self.fc2_value = torch.nn.Linear(self.hidden, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        x = x.reshape(-1, self.reshaped_size)
        x_ac = self.fc1_actor(x)
        x_ac = F.relu(x_ac)
        x_mean = self.fc2_mean(x_ac)

        x_probs = F.softmax(x_mean, dim=-1)
        dist = Categorical(x_probs)

        x_cr = self.fc1_critic(x)
        x_cr = F.relu(x_cr)
        value = self.fc2_value(x_cr)

        return dist, value



class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs):
        super(ActorCritic, self).__init__()
       
        #convolutional part 
        self.conv1 = nn.Conv2d(num_inputs, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)
        #end convolution

        self.fc1 = nn.Linear(self.conv4.out_channels*3*3, 256)
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)
        num_outputs = 3
        #critic
        self.critic_linear = nn.Linear(256, 1)
        #actor
        self.actor_linear = nn.Linear(256, num_outputs)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, inp):
        inp = inp.view(1,1,100,100)
        inp = inp.float()

        x = self.pool(F.relu(self.conv1(inp)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(self.pool(F.relu(self.conv3(x))))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc1(x)
        
        return self.critic_linear(x), self.actor_linear(x)



class Agent(object):
    def __init__(self, env, num_inputs, player_id=1):
    
        if type(env) is not Wimblepong:
            raise TypeError("I'm not a very smart AI. All I can play is Wimblepong.")
        self.env = env
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Set the player id that determines on which side the ai is going to play
        self.actor_critic = ActorCritic(num_inputs)
        self.actor_critic = self.actor_critic.to(self.device)
        #self.optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
        self.player_id = player_id  
        # Ball prediction error, introduce noise such that SimpleAI reflects not
        # only in straight lines
        self.bpe = 4               
        self.name = "Maialo"
       
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=5e-3)
        self.gamma = 0.98
        
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.values = [] # values saved during the training 

    def update(self):
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.device).squeeze(-1)
        values = torch.stack(self.values, dim=0).to(self.device).squeeze(-1) # values from the network
        self.states, self.action_probs, self.rewards, self.values = [], [], [], []

        
        discounted_rewards = self.discount_rewards(rewards,self.gamma) # TASK 3
        
        advantage =  discounted_rewards - values 
        advantage -= torch.mean(advantage)
        advantage /= torch.std(advantage.detach())

        weighted_probs = -action_probs* advantage.detach() 


        actor_loss = weighted_probs.mean()
        critic_loss = advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss 
        
        
        ac_loss.backward()

    
        self.optimizer.step()
        self.optimizer.zero_grad()



    def get_action(self, state, epsilon=0.05):
        state = torch.tensor(state)
        state.to(self.device)
        value, policy = self.actor_critic.forward(state)
        prob = F.softmax(policy, dim=-1)
        log_prob = F.log_softmax(policy, dim=-1)
        #entropy = -(log_prob * prob).sum(1, keepdim=True)
        #entropies.append(entropy)

        action = prob.multinomial(num_samples=1).detach()
        log_prob = log_prob.gather(1, action)
       
        self.action_probs.append(log_prob)
        self.values.append(value)
        return action

    def get_name(self):
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def store_outcome(self, observation, action_taken, reward):
        self.states.append(observation)
        self.rewards.append(torch.Tensor([reward]))

    def reset(self):
        # Nothing to done for now...
        return

    def discount_rewards(self,r, gamma):
        discounted_r = torch.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size(-1))):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r


