import numpy as np

def acrefeet_to_m3(x):
    return x*1233.48

def seconds_to_month(x):
    return x*2592000

def month_to_annual(array):
    return np.add.reduceat(array, np.arange(0, len(array), 12))
