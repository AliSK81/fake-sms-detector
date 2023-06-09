import numpy as np


def iso_tri(x, m, s):
    return max(min((x - m + s) / s, (m - x + s) / s), 0)


def rect_trap(x, m, s):
    return max(min((x - m + s) / s, 1), 0)


def gaussian(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2)


def sigmoid(x, m, s):
    return 1 / (1 + np.exp(-(x - m) / s))


def negation(x):
    return 1 - x
