""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import _pickle as pickle
import gym
import argparse
import wimblepong
import torch, torchvision
from matplotlib import pyplot as plt
from PIL import Image
import cv2

from torch.distributions import Categorical
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--load", type=str, help="Load an agent to continue training")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

debug = False

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?

#batch_size = 2 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  # I = np.array(I,)
  # I = I[::2,::2].mean(axis=-1) # downsample by factor of 2
  # I[I < 50] = 0
  # I[I > 50] = 1
  # I = np.expand_dims(I, axis=-1)
  I = np.array(I, )
  I = cv2.resize(I, (int(80), int(80))).mean(axis=-1)
  I[I < 50] = 0
  I[I > 50] = 1
  #plt.imshow(I)
  #plt.show()
  I = np.expand_dims(I, axis=-1)

  # img = torch.Tensor(I)  # convert to torch Tensor
  # img = img.permute(2, 0, 1)  # fix the shape from (200, 200, 3) to (3, 200, 200)
  # img = transforms.ToPILImage()(img)  # transform Tensor to Image
  # img = transforms.Grayscale(num_output_channels=1)(img)  # Grayscale image (one channel: (1, 200, 200))
  # img = transforms.Resize((40, 40), interpolation=Image.NEAREST)(img)  # resize image to (1, 40, 40)
  # img = transforms.ToTensor()(img)  # back to torch Tensor
  # # img = transforms.Normalize(1,1)(img)
  #
  # mask = img > 0.7
  # img = torch.where(mask, torch.tensor([0.]), img)
  # mask = img > 0.0
  # img = torch.where(mask, torch.tensor([1.]), img)

  #img_numpy = img.numpy()

  return I.astype(np.float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
player_id = 1
opponent_id = 3 - player_id
opponent = wimblepong.SimpleAi(env, opponent_id)

env.set_names('giorgia', opponent.get_name())

observation,_ = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
points =0
list_reward_sum = []
list_running_reward = []
while True:
  if render: env.render()

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)

  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  action1 = 1 if np.random.uniform() < aprob else 2 # roll the dice!
  action2 = opponent.get_action()

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = 1 if action1 == 2 else 0 # a "fake label"
  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  (observation, _), (reward, _), done, info = env.step((action1, action2))
  reward_sum += reward/10.0

  drs.append(reward/10.0) # record reward (has to be done after we call step() to get reward for previous action)
  if done and points<21:
    observation,_ = env.reset() # reset env
    points +=1
  elif done: # an episode finished
    points =0
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs).astype('float64')
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    grad = policy_backward(eph, epdlogp)
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in model.items():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    
    print('resetting env. episode %i reward total was %f. running mean: %f' % (episode_number, reward_sum, running_reward))
    list_reward_sum.append(reward_sum)
    list_running_reward.append(running_reward)
    if episode_number % 100 == 0:
      np.savetxt('reward_sum.txt', np.array(list_reward_sum), delimiter=',')
      np.savetxt('running_reward.txt', np.array(list_running_reward), delimiter=',')
      pickle.dump(model, open('save.p', 'wb'))
    reward_sum = 0
    observation,_ = env.reset() # reset env
    prev_x = None

  if reward != 0 and debug: # Pong has either +1 or -1 reward exactly when game ends.
    print ('ep %d: game finished, reward: %f' % (episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))
