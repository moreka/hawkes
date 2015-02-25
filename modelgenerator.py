import numpy as np


class Events():
    """
    Data structure for holding events
    """

    def __init__(self, users=None, products=None, times=None, count=None):
        self.users = users
        self.times = times
        self.products = products
        self.count = count

    @staticmethod
    def factory(n):
        event_users = np.zeros(n)
        event_products = np.zeros(n)
        event_times = np.ones(n) * (-1)
        return Events(event_users, event_products, event_times, n)

    @staticmethod
    def array_factory(n):
        event_users = [0] * n
        event_products = [0] * n
        event_times = [-1] * n
        return Events(event_users, event_products, event_times, n)