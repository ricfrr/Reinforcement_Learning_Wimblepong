import numpy as np
import matplotlib.pyplot as plt
file_reward = np.loadtxt("reward_history.txt")
file_timesteps = np.loadtxt("timestep_history.txt")
file_point = np.loadtxt("point_history_1.txt")
plt.plot(file_reward)
plt.show()