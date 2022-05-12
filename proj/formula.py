import numpy as np


def compute_t(d, y):
    # a = d - 1
    # b = -2 * (d + 1) / y
    # c = d + 1
    # return (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return ((d + 1) - np.sqrt((d + 1) ** 2 - (d ** 2 - 1) * y)) / ((d - 1) * y)


def compute_sin(t):
    return 2 * t / (1 + t ** 2)


def compute_cos(t):
    return (1 - t ** 2) / (1 + t ** 2)


def compute_left(d, y, t):
    return (d + 1) / (d + compute_cos(t))


def compute_right(d, y, t):
    return y / compute_sin(t)


def compute_assert(d, y):
    t = compute_t(d, y)
    left = compute_left(d, y, t)
    right = compute_right(d, y, t)
    print(left, right)
    assert left == right


compute_assert(0.1, 1)
