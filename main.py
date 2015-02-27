__author__ = 'moreka'

from cascadegenerator import *

import numpy as np
import scipy
import scipy.sparse
import time

n = 50
p = 5
max_alpha = 0.1
max_mu = 0.01

print('Generating sparce alpha and random mu ...')
alpha = max_alpha * scipy.sparse.rand(n, n, 0.3, format='csc').toarray()
mu = np.random.uniform(0, max_mu, (n, p))

print('Generating hawkes process ...')
t1 = time.time()
event_t, event_p, event_u = generate_hawkes(mu, alpha, 1000)

print('Time consumed: ', time.time() - t1)

print(event_t)
