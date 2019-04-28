import random

import numpy as np
import matplotlib.pyplot as plt

mse = lambda A, B: (np.square(A - B)).mean()


def greedy(filters):
    n_filters = len(filters)
    ordered = [filters.pop(random.choice(list(range(12))))]
    while len(ordered) < n_filters:
        focussed_filter = ordered[-1]
        ordered.append(
            filters.pop(min([(i, mse(f, focussed_filter)) for i, f in enumerate(filters)], key=lambda x: x[1])[0]))

    return ordered

def two_opt(filters, improvement_threshold=0.001):  # 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt
    n_filters = len(filters)
    path_distance = lambda r, c: np.sum([mse(c[r[p]], c[r[p - 1]]) for p in range(len(r))])
    # Reverse the order of all elements from element i to element k in array r.
    two_opt_swap = lambda r, i, k: np.concatenate((r[0:i], r[k:-len(r) + i - 1:-1], r[k + 1:len(r)]))
    route = np.arange(n_filters)
    print(route)
    improvement_factor = 1
    best_distance = path_distance(route, filters)  # Calculate the distance of the initial path.
    while improvement_factor > improvement_threshold:  # ROute still improving?
        distance_to_beat = best_distance
        for swap_first in range(1, len(route) - 2):  # From each filter except the first and last,
            for swap_last in range(swap_first + 1, len(route)):  # to each of the filters following,
                new_route = two_opt_swap(route, swap_first, swap_last)  # try reversing the order of these filters
                new_distance = path_distance(new_route,
                                             filters)  # check the total distance with this modification.
                print(new_distance)
                if new_distance < best_distance:  # If the path distance is an improvement,
                    route = new_route  # make this the accepted best route
                    best_distance = new_distance  # and update the distance corresponding to this route.
        improvement_factor = 1 - best_distance / distance_to_beat  # Calculate how much the route has improved.
    return [filters[i] for i in route]  # When the route is no longer improving substantially, stop searching and return the route.