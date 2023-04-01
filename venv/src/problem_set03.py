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
        rewards[i] = soft_policy_iter(F, maxiter, eta_array[i])
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
    np.set_printoptions(threshold=np.inf)
    F = pl_feature(X)
    P = get_transitions(X, A, p, q_low, q_high)
    pi=pi_aggressive(X, A)
    s0=99
    v,a,b=lstd(pi, F,s0, maxiter=10000)
    plt.plot(v)
    plt.show()
    #print(F)
