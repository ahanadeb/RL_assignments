import matplotlib.pyplot as plt
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
    V_pi_lazy = evaluate_analytical(P, p_lazy, R, gamma)
    V_pi_aggr = evaluate_analytical(P, p_arr, R, gamma)
    plt1 = plot_3("Lazy policy values", "Aggressive policy values", "V_lazy - V_aggressive", V_pi_lazy, V_pi_aggr)
    plt1.show()
    print("V_lazy(50): ", V_pi_lazy[49], "\nV_aggressive(50): ", V_pi_aggr[49], "\n")
    print("V_lazy(80): ", V_pi_lazy[79], "\nV_aggressive(80): ", V_pi_aggr[79], "\n")

    # optimal value vs lazy policy values
    V_policy_star, V_policy, opt_policy, time_policy = policy_iteration(X, P, R, A, gamma, max_iter=100)
    V_value_star, V_value, time_value = value_iteration(P, R, gamma, A, X, max_iter=100)

    plt2 = plot_4("10 iterations", "20 iterations", "50 iterations", "100 iterations", V_value[:, 10],
                  V_value[:, 20], V_value[:, 50], V_value[:, 98])
    plt2.show()
    plt3 = plot_4("2 iteration", "10 iterations", "50 iterations", "100 iterations", V_policy[:, 1],
                  V_policy[:, 10], V_policy[:, 50], V_policy[:, 98])
    plt3.show()
    # state comparison
    for i in range(0, 5):
        plt.plot(V_value[24 * i, :], 'b')
        plt.plot(V_policy[24 * i, :], 'r')
    plt.xlabel('number of iterations')
    plt.ylabel('values')
    plt.show()
    plt2 = plot_3("V_pi_lazy values", "V* values", "V*-V_pi_lazy", V_policy_star, V_pi_lazy)
    plt2.show()
    plt4 = plot_3("V_pi_aggressive values", "V* values", "V*-V_pi_aggressive", V_policy_star, V_pi_aggr)
    plt4.show()

    # which is more convenient
    e = 0.1  # tolerance for convergence
    for i in range(0, V_policy.shape[1]):
        if i > 2:
            change = np.sum(V_policy[:, i - 1] - V_policy[:, i])
            if change < e:
                print("Policy iteration converges after ", i, " iterations. Time taken: ", (time_policy / 100) * i,
                      " seconds.\n")
                break

    for i in range(0, V_value.shape[1]):
        if i > 2:
            change = np.sum(V_value[:, i - 1] - V_value[:, i])
            if change < e:
                print("Value iteration converges after ", i, " iterations. Time taken: ", (time_value / 100) * i,
                      " seconds.\n")
                break


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    solve()
