import numpy as np

def evaluate_difference(func, pol, interval, distance=100):
    real_distance = round(distance ** -1, 12)
    lower_bound, upper_bound = interval
    iterator = lower_bound
    lista = []
    while iterator <= upper_bound:
        difference = abs(func(iterator) - pol(iterator))
        iterator = round(iterator + real_distance, 12)
        if (difference == 0): continue
        lista.append((iterator, difference))
    return lista

def median_error(func, pol, interval, distance):
    return np.median([elem[1] for elem in evaluate_difference(func, pol, interval, distance)])

def max_relative_error(func, pol, interval, distance):
    return max([abs(elem[1] / func(elem[0])) for elem in evaluate_difference(func, pol, interval, distance)])
