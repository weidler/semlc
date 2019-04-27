import random

import numpy

mse = lambda A, B: (numpy.square(A - B)).mean()


def greedy(filters):
    n_filters = len(filters)
    ordered = [filters.pop(random.choice(list(range(12))))]
    while len(ordered) < n_filters:
        focussed_filter = ordered[-1]
        ordered.append(
            filters.pop(min([(i, mse(f, focussed_filter)) for i, f in enumerate(filters)], key=lambda x: x[1])[0]))

    return ordered
