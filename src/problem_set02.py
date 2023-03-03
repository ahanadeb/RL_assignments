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

def approx_eval():
    F1 = fine_feature(X)
    V, V_hist, policy = lstd_approx(F1, maxiter=int(103))
    print(len(V_hist))
    print(V_hist[0].shape)
    print(policy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 3))
    x = np.arange(100)
    ax1.plot(x, V_hist[10])
    ax2.plot(x, V_hist[100])
    ax1.title.set_text("10 iterations")
    ax2.title.set_text("100 iterations")
    ax1.set(xlabel='states', ylabel='values')
    ax2.set(xlabel='states', ylabel='values')
    plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 3))
    pi_l = pi_lazy(X, A)
    V4 = lstd(pi_l, F1, maxiter=int(1e+7))
    pi_ar = pi_aggressive(X, A)
    V5 = lstd(pi_ar, F1, maxiter=int(1e+7))
    x = np.arange(100)
    ax1.plot(x, V4)
    ax2.plot(x, V5)
    ax3.plot(x, V_hist[100])
    ax1.title.set_text("Lazy policy evaluation (LSTD)")
    ax2.title.set_text("Aggressive policy evaluation (LSTD)")
    ax3.title.set_text("Approximate policy evaluation (LSTD)")
    ax1.set(xlabel='states', ylabel='values')
    ax2.set(xlabel='states', ylabel='values')
    ax3.set(xlabel='states', ylabel='values')
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 3))
    x = np.arange(100)
    ax1.plot(x, V_hist[100] - V4)
    ax2.plot(x, V_hist[100] - V5)
    ax1.title.set_text("V(Approximate) - V(Lazy)")
    ax2.title.set_text("V(Approximate) - V(Aggressive)")
    ax1.set(xlabel='states', ylabel='values')
    ax2.set(xlabel='states', ylabel='values')
    plt.show()

    return V, V_hist, policy



def problem_set02():
    td0_eval()
    # lstd_eval()
    # approx_eval()



    return 0
