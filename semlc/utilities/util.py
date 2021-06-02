import math
import os
import sys
from typing import Tuple


def closest_factors(number: int) -> Tuple[int, int]:
    """Return the two integers that are factors of given number and closest together."""
    test_num = int(math.sqrt(number))
    while number % test_num != 0:
        test_num -= 1

    return test_num, int(number / test_num)


class HiddenPrints:
    """Context that hides print calls."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


if __name__ == '__main__':
    print(closest_factors(32))
