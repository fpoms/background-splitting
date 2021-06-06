import time

class EMA():
    '''Exponential moving average'''
    def __init__(self, init: float):
        self.value = init
        self.n_samples = 0

    def __iadd__(self, other: float):
        self.n_samples += 1
        self.value = (
            self.value
            - (self.value / self.n_samples)
            + other / self.n_samples)
        return self
