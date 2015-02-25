import numpy as np


class Events():
    """
    Data structure for holding events
    """

    def __init__(self, users=None, products=None, times=None):
        self.users = users
        self.times = times
        self.products = products

    @staticmethod
    def factory(n):
        event_users = np.zeros(n)
        event_products = np.zeros(n)
        event_times = np.ones(n) * (-1)
        return Events(event_users, event_products, event_times)