import math
from typing import Tuple


def closest_factors(number: int) -> Tuple[int, int]:
    """Return the two integers that are factors of given number and closest together."""
    test_num = int(math.sqrt(number))
    while number % test_num != 0:
        test_num -= 1

    return test_num, int(number / test_num)


if __name__ == '__main__':
    print(closest_factors(32))
