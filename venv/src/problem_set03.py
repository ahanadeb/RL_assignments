from utils.feature_space import *
import matplotlib.pyplot as plt
import numpy as np
from utils.util_functions import *
from utils.reward import *
from utils.transition import get_transitions
from utils.policies import *
from utils.plot import *
from utils.params import *
from utils.policy_eval import *
from utils.approx_policy_iter import *
from utils.line_plot import *
from utils.td0_eval import *
from utils.lstd_eval import *
from utils.soft_policy_iteration import *
from tqdm import tqdm

def problem_set03():
    F = pl_feature(X)
    M = 10
    eta_array = np.logspace(-2, +2, num=M)
    rewards = np.zeros((len(eta_array), 1))
    maxiter = 100
    for i in tqdm(range(M)):
       # rewards[i] = soft_policy_iter(F, maxiter, eta_array[i])
        rewards[i] = soft_policy_iter(F, maxiter, eta_array)
        print(rewards[i])
    x = np.arange(M)
    print(rewards)
    plt.semilogx(eta_array, rewards)
    max_reward = np.argmax(rewards)
    opt_eta = eta_array[max_reward]

    plt.ylabel('rewards gathered')
    plt.xlabel('eta values (log scale)')
    plt.axvline(x=opt_eta, color='r')
    plt.figtext(.6, .8, "optimal eta = "+ "{:.3f}".format(opt_eta))
    plt.show()
    return plt

def check():
    F = pl_feature(X)
    M = 100
    eta_array = np.logspace(-2, +2, num=M)
    #eta_array = np.zeros((M,1))+80
    rewards = np.zeros((len(eta_array), 1))
    maxiter = 100

    rewards1,rewards2 = soft_policy_iter(F, maxiter, eta_array)

    x = np.arange(M)
    print(rewards)
    plt.plot(x, rewards2)
    max_reward = np.argmax(rewards)
    opt_eta = eta_array[max_reward]

    plt.ylabel('rewards gathered')
    plt.xlabel('iterations')
   # plt.axvline(x=opt_eta, color='r')
   # plt.figtext(.6, .8, "optimal eta = " + "{:.3f}".format(opt_eta))
    plt.show()
    return plt
