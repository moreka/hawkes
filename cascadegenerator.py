from __future__ import print_function

import numpy as np
import scipy
import scipy.sparse
import time
import math

from modelgenerator import Events


def hawkes_intensity(mu, alpha, events, t):
    intensity = mu.copy()

    for i in range(events.count):
        ti = events.times[i]
        if ti >= t:
            break

        pi = events.products[i]
        ui = events.users[i]
        intensity[:, pi] = intensity[:, pi] + alpha[ui, :] * math.exp(ti - t)

    return intensity


def get_random_user(intensity, intensity_sum):
    row_sum = intensity.sum(axis=1)
    random_picked_user = np.random.multinomial(1, row_sum / intensity_sum)
    return np.flatnonzero(random_picked_user)[0]


def get_random_product(intensity):
    col_sum = np.exp(intensity).sum(axis=0)
    random_picked_product = np.random.multinomial(1, col_sum / col_sum.sum())
    return np.flatnonzero(random_picked_product)[0]


def generate_hawkes(mu, alpha, n, t0=0, intensity=hawkes_intensity):

    events = Events.factory(n)

    k = 0
    t = t0

    while k < n:
        intensity_at_t = intensity(mu, alpha, events, t)

        m_t = intensity_at_t.sum()
        s = np.random.exponential(1. / m_t)
        u = np.random.uniform()

        intensity_at_t_s = intensity(mu, alpha, events, t + s)

        t = t + s

        if u * m_t < intensity_at_t_s.sum():

            user = get_random_user(intensity_at_t, m_t)
            product = get_random_product(intensity_at_t)

            events.users[k] = user
            events.products[k] = product
            events.times[k] = t

            k += 1

            if k % 10 == 0:
                print('\tnow at iteration ', k)

    return events


def main():
    n = 50
    p = 5
    max_alpha = 0.1
    max_mu = 0.01

    print('Generating sparce alpha and random mu ...')
    alpha = max_alpha * scipy.sparse.rand(n, n, 0.3, format='csc').toarray()
    mu = np.random.uniform(0, max_mu, (n, p))

    print('Generating hawkes process ...')
    t1 = time.time()
    events = generate_hawkes(mu, alpha, 1000)

    print('Time consumed: ', time.time() - t1)

    print(events.times)

if __name__ == '__main__':
    main()