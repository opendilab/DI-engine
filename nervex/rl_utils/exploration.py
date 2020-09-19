import math


def epsilon_greedy(start, end, decay):
    return lambda x: (start - end) * math.exp(-1 * x / decay) + end
