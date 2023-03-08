import matplotlib.pyplot as plt
import numpy as np
from utils.util_functions import *
from utils.reward import *
from utils.transition import get_transitions
from utils.policies import *
from utils.plot import *
from utils.params import *
import random


def TD0(policy, F, a1, b, maxiter):
    print("wokring")
    reward = get_reward(X, A)
    P = get_transitions(X, A, p, q_low, q_high)
    D = F.shape[0]
    theta = np.zeros((D, 1))
    P_pi = trans(P, policy)
    # loop for each episode here. Here episode = 1
    s = 99  # initial state
    states = (np.arange(X)).reshape((X,))
    for i in range(maxiter):
        alpha = a1 / (i + b)
        # get action at this state
        a = np.argmax(policy[s, :])
        r = reward[s, a]
        next_s = random.choices(states, weights=P_pi[s, :].reshape((X,)), k=1)[0]
        delta_t = r + gamma * np.matmul(theta.T, F[:, next_s])[0] - np.matmul(theta.T, F[:, s])[0]
        theta = theta + (alpha * delta_t * F[:, s]).reshape((D, 1))
        s = next_s
        if i == 1e+4:
            V1 = np.transpose(np.matmul(theta.T, F))
        if i == 1e+5:
            V2 = np.transpose(np.matmul(theta.T, F))
        if i == 1e+6:
            V3 = np.transpose(np.matmul(theta.T, F))
        if i == 1e+7-1:
            V4 = np.transpose(np.matmul(theta.T, F))
    return V1,V2,V3,V4


def lstd(policy, F, maxiter):
    sigma = 1e-5
    D = F.shape[0]
    A_mat = np.eye((F.shape[0])) + sigma
    b_mat = np.zeros((F.shape[0], 1))
    reward = get_reward(X, A)
    P = get_transitions(X, A, p, q_low, q_high)

    theta = np.zeros((X, 1))
    P_pi = trans(P, policy)
    # loop for each episode here. Here episode = 1
    s = 99  # initial state
    states = (np.arange(X)).reshape((X,))
    for i in range(maxiter):
        # get action at this state
        a = np.argmax(policy[s, :])
        r = reward[s, a]
        next_s = random.choices(states, weights=P_pi[s, :].reshape((X,)), k=1)[0]
        x = np.matmul((F[:, s].reshape((D,1))), np.transpose((F[:, s].reshape((D,1)) - gamma * F[:, next_s].reshape((D,1)))))
        A_mat = A_mat + np.matmul((F[:, s].reshape((D,1))), np.transpose((F[:, s].reshape((D,1)) - gamma * F[:, next_s].reshape((D,1)))))
        b_mat = b_mat + r * F[:, s].reshape((b_mat.shape[0], 1))
        s = next_s
        theta = np.matmul(np.linalg.inv(A_mat), b_mat)
    return np.transpose(np.matmul(theta.T, F))
