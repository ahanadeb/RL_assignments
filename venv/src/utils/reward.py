import numpy as np


def get_reward(X, A):
    r = np.zeros((X,A))
    for i in range(0,X):
        for j in range(0,2):
            cost = cost_of_action(j)
            r[i,j] = -np.power((i / X), 2) - cost

    return r


def cost_of_action(a):
    cost = 0
    if a == 1:
        cost = 0.05
    return cost
