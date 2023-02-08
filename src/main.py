import numpy as np
from utils.util_functions import *
from utils.reward import *
from utils.transition import get_transitions
from utils.policies import *
from utils.plot import *

X = 100  # state space, length of queue
gamma = 0.9  # discount factor
A = 2  # number of actions
p = 0.5
q_low = 0.51
q_high = 0.6


def solve():
    print(f'Hi,')
    P = get_transitions(X, A, p, q_low, q_high)
    p_lazy = pi_lazy(X, A)
    p_arr = pi_aggressive(X, A)
    R = get_reward(X, A)
    # lazy policy vs aggressive policy
    V_pi_lazy = evaluate_power_iter(P, p_lazy, R, gamma)
    V_pi_aggr = evaluate_power_iter(P, p_arr, R, gamma)
    plt1 = plot_3("Lazy policy values", "Aggressive policy values", "V_lazy - V_aggressive", V_pi_lazy, V_pi_aggr)
    plt1.show()
    print("V_lazy(50): ", V_pi_lazy[51], "\nV_aggressive(50): ", V_pi_aggr[51], "\n")
    print("V_lazy(80): ", V_pi_lazy[81], "\nV_aggressive(80): ", V_pi_aggr[81], "\n")

    # optimal value vs lazy policy values
    V1_star = policy_iteration(X, P, R, A, gamma, max_iter=1000)
    V2_star = value_iteration(P, R, gamma, A, X, max_iter=10)
    plt = plot_3("VI lazy values", "VI opt values", "V_pi_lazy-V_opt", V_pi_lazy, V1_star)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    solve()
