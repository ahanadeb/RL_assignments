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


def lstd_approx(F, maxiter):
    Q_est = np.zeros((X,A))
    policy = np.zeros((X, A))+.5
    r = get_reward(X, A)
    V_hist=[]
    for i in range(0,maxiter):
        V = lstd(policy, F, maxiter=int(1e+5)) #shape (Xx1)
        for x in range(0,X):
            for a in range(0,A):
                if x == 0:
                    l = x
                    u=x+1
                if x == X-1:
                    l = x-1
                    u = x
                else:
                    l = x-1
                    u = x+1
                Q_est[x,a]= r[x,a]+ gamma*(1-p)*(q[a]*V[l,0] + (1-q[a])*V[x,0]) \
                                +gamma*p*(q[a]*V[x,0] + (1-q[a])*V[u,0])
            policy[x] = np.zeros((A))
            policy[x, np.argmax(Q_est[x])] = 1
        V_hist.append(V)
    return V, V_hist, policy