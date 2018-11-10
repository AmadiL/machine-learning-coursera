import functools
import numpy as np

def print_name(f):
    @functools.wraps(f)
    def wrapper(*args, **kwds):
        print('\nRunning {}() ...'.format(f.__name__))
        return f(*args, **kwds)
    return wrapper

def pause_after(f):
    @functools.wraps(f)
    def wrapper(*args, **kwds):
        r = f(*args, **kwds)
        input("Press Enter to continue...")
        return r
    return wrapper

def sigmoid(x):
    return 1 / (1 + np.exp(-x))