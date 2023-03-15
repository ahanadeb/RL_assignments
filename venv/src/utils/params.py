import numpy as np
X = 100  # state space, length of queue
gamma = 0.99  # discount factor
A = 2  # number of actions
p = 0.5
q_low = 0.51
q_high = 0.6
q = [q_low, q_high]
