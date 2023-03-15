import matplotlib.pyplot as plt
import numpy as np
from utils.util_functions import *
from utils.reward import *
from utils.transition import get_transitions
from utils.policies import *
from utils.plot import *
from utils.params import *
import random
from utils.policy_eval import *
from tqdm import tqdm


def soft_policy_iter(F, maxiter, eta):
    Q_est = np.zeros((X, A))
    policy = pi_uniform(X, A)
    new_policy = np.zeros((X, A))
    r = get_reward(X, A)
    total_reward = 0
    s0 = 99  # starting with a full queue

    for i in range(0,maxiter):
        V, s0, r_lstd = lstd(policy, F, s0, maxiter=int(1e+5))
        total_reward = total_reward + r_lstd
        for x in range(0, X):
            for a in range(0, A):
                if x == 0:
                    l = x
                    u = x + 1
                if x == X - 1:
                    l = x - 1
                    u = x
                else:
                    l = x - 1
                    u = x + 1
                Q_est[x, a] = r[x, a] + gamma * (1 - p) * (q[a] * V[l, 0] + (1 - q[a]) * V[x, 0]) \
                              + gamma * p * (q[a] * V[x, 0] + (1 - q[a]) * V[u, 0])
            Q_est[x, :] = Q_est[x, :] - np.max(Q_est[x, :])
            for a in range(0, A):
                new_policy[x, a] = policy[x, a] * np.exp(eta * Q_est[x, a])
            new_policy[x, :] = new_policy[x, :] / np.sum(new_policy[x, :])
        policy = new_policy

    return total_reward
