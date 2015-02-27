from __future__ import print_function, division

cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport exp


DTYPE = np.double
ctypedef np.double_t DTYPE_t

ITYPE = np.int
ctypedef np.int_t ITYPE_t

LTYPE = np.long
ctypedef np.long_t LTYPE_t


@cython.boundscheck(False)
cdef intensity(np.ndarray[DTYPE_t, ndim=2] mu,
               np.ndarray[DTYPE_t, ndim=2] alpha,
               np.ndarray[DTYPE_t] event_times,
               np.ndarray[ITYPE_t] event_users,
               np.ndarray[ITYPE_t] event_prods,
               double t):

    cdef np.ndarray[DTYPE_t, ndim=2] intens
    cdef unsigned int i, pi, ui
    cdef double ti

    intens = 1 * mu

    for i in range(event_times.shape[0]):
        ti = event_times[i]
        if ti >= t:
            break

        pi = event_prods[i]
        ui = event_users[i]
        intens[:, pi] = intens[:, pi] + alpha[ui, :] * exp(ti - t)

    return intens

cdef int get_random_user(np.ndarray[DTYPE_t, ndim=2] intensity, double intensity_sum):
    cdef np.ndarray[DTYPE_t] row_sum
    cdef np.ndarray[LTYPE_t] random_picked_user

    row_sum = intensity.sum(axis=1)
    random_picked_user = np.random.multinomial(1, row_sum / intensity_sum)

    return np.flatnonzero(random_picked_user)[0]

cdef int get_random_product(np.ndarray[DTYPE_t, ndim=2] intensity):
    cdef np.ndarray[DTYPE_t] col_sum
    cdef np.ndarray[LTYPE_t] random_picked_product

    col_sum = np.exp(intensity).sum(axis=0)
    random_picked_product = np.random.multinomial(1, col_sum / col_sum.sum())
    return np.flatnonzero(random_picked_product)[0]

def generate_hawkes(np.ndarray[DTYPE_t, ndim=2] mu, 
                    np.ndarray[DTYPE_t, ndim=2] alpha, 
                    int n, 
                    double t0=0):

    cdef np.ndarray[ITYPE_t] event_users = np.zeros(n, dtype=np.int)
    cdef np.ndarray[ITYPE_t] event_prods = np.zeros(n, dtype=np.int)
    cdef np.ndarray[DTYPE_t] event_times = np.ones(n) * (-1)

    cdef int k = 0, user, product
    cdef double t = t0, s, u, m_t
    cdef np.ndarray[DTYPE_t, ndim=2] intensity_at_t, intensity_at_t_s

    while k < n:
        intensity_at_t = intensity(mu, alpha, event_times, event_users, event_prods, t)

        m_t = intensity_at_t.sum()
        s = np.random.exponential(1. / m_t)
        u = np.random.uniform()

        intensity_at_t_s = intensity(mu, alpha, event_times, event_users, event_prods, t + s)

        t = t + s

        if u * m_t < intensity_at_t_s.sum():

            user = get_random_user(intensity_at_t, m_t)
            product = get_random_product(intensity_at_t)

            event_users[k] = user
            event_prods[k] = product
            event_times[k] = t

            k += 1

            if k % 10 == 0:
                print('\tnow at iteration ', k)

    return event_times, event_prods, event_users

