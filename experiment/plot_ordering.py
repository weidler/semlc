import random
import matplotlib.pyplot as plt

random.seed(111)

original_ordering = list(range(1, 64))
inhibited_ordering = original_ordering.copy()
random.shuffle(inhibited_ordering)

plt.plot(original_ordering, inhibited_ordering, "--", linewidth=1)

min_sequence_length = 3

current_sequence = [inhibited_ordering[0]]
current_sequence_xs = [1]
current_direction = -1
for i, position in enumerate(inhibited_ordering[1:], 2):
    direction = position - current_sequence[-1]
    # check if same direction
    if ((direction == current_direction) and (direction == 0)) or (direction * current_direction > 0):
        current_sequence.append(position)
        current_sequence_xs.append(i)
    else:
        if len(current_sequence) >= min_sequence_length:
            plt.plot(current_sequence_xs, current_sequence, color="red" if current_direction == -1 else "green")
        current_sequence = [current_sequence[-1], position]
        current_sequence_xs = [current_sequence_xs[-1], i]
        current_direction = direction / abs(direction)

plt.scatter(original_ordering, inhibited_ordering)
plt.show()
