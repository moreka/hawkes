from __future__ import print_function

import numpy as np
import cvxpy as cvx
from math import log, exp
from functools import partial
import scipy
import time
from cascadegenerator import *


def exp_sum(arr, row):
    return np.exp(arr[row, :]).sum()


def log_likelihood(event_t, event_u, event_p, mu, alpha, T):
    likelihood = -T * mu.sum()
    user_sums = alpha.sum(axis=1)

    _intensity = partial(intensity, mu, alpha, event_t, event_u, event_p)
    for j in range(event_t.shape[0]):
        t_j = event_t[j]
        if t_j < T:
            intens = _intensity(t_j)
            u_j = event_u[j]
            p_j = event_p[j]
            likelihood += log(intens.sum(axis=1)[u_j])
            likelihood += intens[u_j, p_j]
            likelihood += -(log(exp_sum(intens, u_j)))

            # integral
            likelihood -= user_sums[u_j] * (1 - exp(t_j - T))

        else:
            break
    return likelihood


N = 1000
n = 5
p = 1
max_alpha = 0.1
max_mu = 0.01


def generate_hawkes_process():
    print('Generating sparce alpha and random mu ...')
    alpha = max_alpha * scipy.sparse.rand(n, n, 0.3, format='csc').toarray()
    mu = np.random.uniform(0, max_mu, (n, p))

    print('Generating hawkes process ...')
    t1 = time.time()
    event_t, event_p, event_u = generate_hawkes(mu, alpha, N)

    print('Time consumed: ', time.time() - t1)
    print('Given MU: ', mu)
    print('Given Alpha: ', alpha)
    return event_t, event_p, event_u


if __name__ == '__main__':
    # np.random.seed(1)
    events_t, events_p, events_u = generate_hawkes_process()
    T = events_t[-1]

    mu = cvx.Variable(n, p)
    alpha = cvx.Variable(n, n)

    objective = cvx.Maximize(log_likelihood(events_t, events_u, events_p, mu, alpha, T))
    constraints = [0 <= alpha, 0 <= mu, alpha <= 1, mu <= 1]
    prob = cvx.Problem(objective, constraints)

    print(prob.solve(verbose=True, solver='CVXOPT'))
    print(mu.value, alpha.value)
