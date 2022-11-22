import doctest
import math
import itertools
import random
import datetime
from typing import List
from noise import pnoise2
import pytest


def Q_generate_prime(N: int = 10**8) -> int:
    bases = range(2, 20000)
    # This needs to be more complicated
    p = 1
    while any(pow(base, p - 1, p) != 1 for base in bases):
        p = random.SystemRandom().randrange(N)
    return p


def Q_multiplicative_inverse(modulus: int, value: int) -> int:
    """
    >>> Q_multiplicative_inverse(191, 138)
    18
    >>> 18 * 138 % 191
    1
    """
    x = 0
    lastx = 1
    a = modulus
    b = value
    while b:
        a, q, b = b, a // b, a % b
        x, lastx = lastx - q * x, x
    result = (1 - lastx * modulus) // value
    return result + modulus if result < 0 else result


def Q_generate_keys(N: int):
    """
    >>> pubkey, privatekey = Q_generate_keys(2 ** 64)
    >>> msg = 123456789012345
    >>> coded = pow(msg, 65537, pubkey)
    >>> coded != msg
    True
    >>> plain = pow(coded, privatekey, pubkey)
    >>> msg == plain
    True
    """
    # http://en.wikipedia.org/wiki/RSA
    prime1: int = Q_generate_prime(N)
    prime2: int = Q_generate_prime(N)
    totient: int = (prime1 - 1) * (prime2 - 1)
    return prime1 * prime2, Q_multiplicative_inverse(totient, 65537)


def Q_buckets(number_of_items: int, number_of_buckets: int):
    """
    >>> stuff = [_ for _ in range(1, 53)]
    >>> for start, end in Q_buckets(52, 12):
    ...    print(start, end, stuff[start:end])
    ...
    0 5 [1, 2, 3, 4, 5]
    5 10 [6, 7, 8, 9, 10]
    10 15 [11, 12, 13, 14, 15]
    15 20 [16, 17, 18, 19, 20]
    20 24 [21, 22, 23, 24]
    24 28 [25, 26, 27, 28]
    28 32 [29, 30, 31, 32]
    32 36 [33, 34, 35, 36]
    36 40 [37, 38, 39, 40]
    40 44 [41, 42, 43, 44]
    44 48 [45, 46, 47, 48]
    48 52 [49, 50, 51, 52]
    """
    items_per_bucket = number_of_items // number_of_buckets
    leftover = number_of_items % number_of_buckets
    leftover_per_bucket = max(leftover // number_of_buckets, 1) if leftover > 0 else 0
    starting_point = 0
    ending_point = 0
    while ending_point <= number_of_items and number_of_buckets > 0:
        starting_point = ending_point
        ending_point = min(ending_point + items_per_bucket + leftover_per_bucket, number_of_items)
        items_remaining = number_of_items - ending_point
        number_of_buckets -= 1
        if number_of_buckets:
            leftover = items_remaining % number_of_buckets
            leftover_per_bucket = max(leftover // number_of_buckets, 1) if leftover > 0 else 0
        yield starting_point, ending_point


def Q_clamp(value: float, minimum_limit: float, maximum_limit: float):
    """
    >>> Q_clamp(value=16.5, minimum_limit=-29.1, maximum_limit=142.6)
    16.5
    >>> Q_clamp(value=142.5, minimum_limit=62.7, maximum_limit=100.0)
    100.0
    >>> Q_clamp(value=-0.5, minimum_limit=-0.4, maximum_limit=1.0)
    -0.4
    """

    # if value < minimum_limit:
    #     return minimum_limit
    # if value > maximum_limit:
    #     return maximum_limit
    # return value
    return max(min(value, maximum_limit), minimum_limit)


def Q_sat(value: float):
    """
    >>> Q_sat(0.5)
    0.5
    >>> Q_sat(1.5)
    1.0
    >>> Q_sat(-0.5)
    0.0
    """
    return Q_clamp(value=value, minimum_limit=0.0, maximum_limit=1.0)


# """
#  █████       ██████████ █████ █████      █████████     ███████    ██████   ██████ ███████████  █████ ██████   █████   █████████   ███████████ █████    ███████    ██████   █████  █████████
# ░░███       ░░███░░░░░█░░███ ░░███      ███░░░░░███  ███░░░░░███ ░░██████ ██████ ░░███░░░░░███░░███ ░░██████ ░░███   ███░░░░░███ ░█░░░███░░░█░░███   ███░░░░░███ ░░██████ ░░███  ███░░░░░███
#  ░███        ░███  █ ░  ░░███ ███      ███     ░░░  ███     ░░███ ░███░█████░███  ░███    ░███ ░███  ░███░███ ░███  ░███    ░███ ░   ░███  ░  ░███  ███     ░░███ ░███░███ ░███ ░███    ░░░
#  ░███        ░██████     ░░█████      ░███         ░███      ░███ ░███░░███ ░███  ░██████████  ░███  ░███░░███░███  ░███████████     ░███     ░███ ░███      ░███ ░███░░███░███ ░░█████████
#  ░███        ░███░░█      ███░███     ░███         ░███      ░███ ░███ ░░░  ░███  ░███░░░░░███ ░███  ░███ ░░██████  ░███░░░░░███     ░███     ░███ ░███      ░███ ░███ ░░██████  ░░░░░░░░███
#  ░███      █ ░███ ░   █  ███ ░░███    ░░███     ███░░███     ███  ░███      ░███  ░███    ░███ ░███  ░███  ░░█████  ░███    ░███     ░███     ░███ ░░███     ███  ░███  ░░█████  ███    ░███
#  ███████████ ██████████ █████ █████    ░░█████████  ░░░███████░   █████     █████ ███████████  █████ █████  ░░█████ █████   █████    █████    █████ ░░░███████░   █████  ░░█████░░█████████
# ░░░░░░░░░░░ ░░░░░░░░░░ ░░░░░ ░░░░░      ░░░░░░░░░     ░░░░░░░    ░░░░░     ░░░░░ ░░░░░░░░░░░  ░░░░░ ░░░░░    ░░░░░ ░░░░░   ░░░░░    ░░░░░    ░░░░░    ░░░░░░░    ░░░░░    ░░░░░  ░░░░░░░░░
# """


# def Q_get_combinations(array: list, number_of_items: int, selection: list = list()):
#     if number_of_items <= 0:
#         combinations.append(list(selection))
#         print(f'selection {selection}')
#         return

#     for idx in range(len(array) - number_of_items + 1):
#         new_array = array[idx + 1 :]
#         Q_get_combinations(array=new_array, number_of_items=number_of_items - 1, selection=selection + [array[idx]])


# global combinations
# combinations = list()

# def Q_get_lex_combinations(array: list, number_of_items: int, selection: list = list()) -> list:
#     if number_of_items == 0:
#         combinations.append(selection)
#         return

#     for idx in range(len(array)):
#         Q_get_lex_combinations(
#             array=array[:idx] + array[idx + 1 :],
#             number_of_items=number_of_items - 1,
#             selection=selection + [array[idx]],
#         )


def Q_get_lex_combinations_generator(array: list, number_of_items: int, selection: list = list()) -> list:
    """
    >>> for i in range(9):
    ...     points = [x + 1 for x in range(i + 1)]
    ...     print(sum(1 for _ in Q_get_lex_combinations_generator(array=sorted(points), number_of_items=len(points))))
    1
    2
    6
    24
    120
    720
    5040
    40320
    362880
    """
    if number_of_items == 0:
        yield selection

    for idx in range(len(array)):
        yield from Q_get_lex_combinations_generator(
            array=array[:idx] + array[idx + 1 :],
            number_of_items=number_of_items - 1,
            selection=selection + [array[idx]],
        )


def Q_weighted_choice(list_of_choices: list, number_of_choices: int = 1, replacement: bool = False):
    """
    >>> possibilities = [(0, 0.35), (1, 0.15), (2, 0.25), (3, 0.25)]
    >>> results = {}
    >>> test_iterations = 50_000
    >>> for _ in range(test_iterations):
    ...     i = Q_weighted_choice(possibilities, number_of_choices=4, replacement=True)
    ...     results[i[0]] = results.get(i[0], 0) + 1
    ...     results[i[1]] = results.get(i[1], 0) + 1
    ...     results[i[2]] = results.get(i[2], 0) + 1
    ...     results[i[3]] = results.get(i[3], 0) + 1
    >>> for k in sorted(results):
    ...     print(k, possibilities[k][1], abs(possibilities[k][1] - (results[k] / (test_iterations * 4.0))) < 0.01, results[k] / (test_iterations * 4.0)) # doctest: +ELLIPSIS
    0 0.35 True ...
    1 0.15 True ...
    2 0.25 True ...
    3 0.25 True ...

    >>> possibilities = [(0, 0.35), (1, 0.15), (2, 0.25), (3, 0.25)]
    >>> del results
    >>> results = {}
    >>> test_iterations = 50_000
    >>> for _ in range(test_iterations):
    ...     i = Q_weighted_choice(possibilities, number_of_choices=4, replacement=False)
    ...     results[i[0]] = results.get(i[0], 0) + 1
    ...     results[i[1]] = results.get(i[1], 0) + 1
    ...     results[i[2]] = results.get(i[2], 0) + 1
    ...     results[i[3]] = results.get(i[3], 0) + 1
    >>> for k in sorted(results):
    ...     print(k, possibilities[k][1], abs(possibilities[k][1] - (results[k] / (test_iterations * 4.0))) < 0.01, results[k] / (test_iterations * 4.0)) # doctest: +ELLIPSIS
    0 0.35 False ...
    1 0.15 False ...
    2 0.25 True ...
    3 0.25 True ...
    """
    all_choices = list(list_of_choices)
    results = []
    for _ in range(number_of_choices):
        total_weight = sum(weight for item, weight in all_choices)
        random_weight = random.uniform(0, total_weight)
        upto = 0
        for item, weight in all_choices:
            upto += weight
            if upto >= random_weight:
                results.append(item)
                if not replacement:
                    all_choices.remove((item, weight))
                break
    return results


def Q_weighted_choice2(list_of_choices: list, number_of_choices: int = 1, replacement: bool = False):
    """
    >>> possibilities = [(0, 0.35), (1, 0.15), (2, 0.25), (3, 0.25)]
    >>> results = {}
    >>> test_iterations = 50_000
    >>> for _ in range(test_iterations):
    ...     i = Q_weighted_choice2(possibilities, number_of_choices=4, replacement=True)
    ...     results[i[0]] = results.get(i[0], 0) + 1
    ...     results[i[1]] = results.get(i[1], 0) + 1
    ...     results[i[2]] = results.get(i[2], 0) + 1
    ...     results[i[3]] = results.get(i[3], 0) + 1
    >>> for k in sorted(results):
    ...     print(k, possibilities[k][1], abs(possibilities[k][1] - (results[k] / (test_iterations * 4.0))) < 0.01, results[k] / (test_iterations * 4.0)) # doctest: +ELLIPSIS
    0 0.35 True ...
    1 0.15 True ...
    2 0.25 True ...
    3 0.25 True ...

    >>> possibilities = [(0, 0.35), (1, 0.15), (2, 0.25), (3, 0.25)]
    >>> del results
    >>> results = {}
    >>> test_iterations = 50_000
    >>> for _ in range(test_iterations):
    ...     i = Q_weighted_choice2(possibilities, number_of_choices=4, replacement=False)
    ...     results[i[0]] = results.get(i[0], 0) + 1
    ...     results[i[1]] = results.get(i[1], 0) + 1
    ...     results[i[2]] = results.get(i[2], 0) + 1
    ...     results[i[3]] = results.get(i[3], 0) + 1
    >>> for k in sorted(results):
    ...     print(k, possibilities[k][1], abs(possibilities[k][1] - (results[k] / (test_iterations * 4.0))) < 0.01, results[k] / (test_iterations * 4.0)) # doctest: +ELLIPSIS
    0 0.35 False ...
    1 0.15 False ...
    2 0.25 True ...
    3 0.25 True ...
    """
    all_choices = list(list_of_choices)
    results = []
    total_weight = sum(weight for item, weight in all_choices)
    random_weight = random.uniform(0, total_weight)
    upto = 0
    for idx in range(len(all_choices)):
        upto += all_choices[idx][1]
        if upto >= random_weight:
            item = all_choices.pop(idx) if not replacement else all_choices[idx]
            results.append(item[0])
            break
    if number_of_choices > 1:
        results.extend(
            iter(
                Q_weighted_choice2(
                    list_of_choices=all_choices,
                    number_of_choices=number_of_choices - 1,
                    replacement=replacement,
                )
            )
        )

    return results


def Q_weighted_choice3(
    list_of_choices: list, list_of_weights: list, number_of_choices: int = 1, replacement: bool = False
):
    """
    >>> possibilities = [0, 1, 2, 3]
    >>> weights = [0.35, 0.15, 0.25, 0.25]
    >>> results = {}
    >>> test_iterations = 50_000
    >>> for _ in range(test_iterations):
    ...     i = Q_weighted_choice3(list_of_choices=possibilities, list_of_weights=weights, number_of_choices=4, replacement=True)
    ...     results[i[0]] = results.get(i[0], 0) + 1
    ...     results[i[1]] = results.get(i[1], 0) + 1
    ...     results[i[2]] = results.get(i[2], 0) + 1
    ...     results[i[3]] = results.get(i[3], 0) + 1
    >>> for k in sorted(results):
    ...     print(k, weights[k], abs(weights[k] - (results[k] / (test_iterations * 4.0))) < 0.01, results[k] / (test_iterations * 4.0)) # doctest: +ELLIPSIS
    0 0.35 True ...
    1 0.15 True ...
    2 0.25 True ...
    3 0.25 True ...

    >>> possibilities = [0, 1, 2, 3]
    >>> weights = [0.35, 0.15, 0.25, 0.25]
    >>> del results
    >>> results = {}
    >>> test_iterations = 50_000
    >>> for _ in range(test_iterations):
    ...     i = Q_weighted_choice3(list_of_choices=possibilities, list_of_weights=weights, number_of_choices=4, replacement=False)
    ...     results[i[0]] = results.get(i[0], 0) + 1
    ...     results[i[1]] = results.get(i[1], 0) + 1
    ...     results[i[2]] = results.get(i[2], 0) + 1
    ...     results[i[3]] = results.get(i[3], 0) + 1
    >>> for k in sorted(results):
    ...     print(k, weights[k], abs(weights[k] - (results[k] / (test_iterations * 4.0))) < 0.01, results[k] / (test_iterations * 4.0)) # doctest: +ELLIPSIS
    0 0.35 False ...
    1 0.15 False ...
    2 0.25 True ...
    3 0.25 True ...
    """
    all_choices = list(list_of_choices)
    all_weights = list(list_of_weights)
    assert len(all_choices) == len(all_weights)
    results: list = list()
    for _ in range(number_of_choices):
        total_weight = sum(all_weights)
        random_weight = random.uniform(0, total_weight)
        upto = 0
        for idx in range(len(all_choices)):
            upto += all_weights[idx]
            if upto >= random_weight:
                if not replacement:
                    item = all_choices.pop(idx)
                    all_weights.pop(idx)
                else:
                    item = all_choices[idx]
                results.append(item)
                break
    return results


class Q_Vector2D:
    def __init__(self, angle: float, magnitude: float):
        self.angle = angle
        self.magnitude = magnitude

    def __str__(self):
        return f'Q_Vector2D(angle={self.angle}, magnitude={self.magnitude}, x={self.x}, y={self.y})'

    def __repr__(self):
        return f'Q_Vector2D(angle={self.angle}, magnitude={self.magnitude})'

    def __add__(self, other):
        """
        >>> PI = math.pi
        >>> v1 = Q_Vector2D(angle=PI, magnitude=10)
        >>> v2 = Q_Vector2D(angle=PI, magnitude=0)
        >>> v3 = v1 + v2
        >>> print(v3)
        Q_Vector2D(angle=3.141592653589793, magnitude=10, x=-10.0, y=1.2246467991473533e-15)
        >>> v3.x
        -10.0
        >>> v3.y
        1.2246467991473533e-15
        """
        if self.magnitude == 0 and other.magnitude == 0:
            return self
        if self.magnitude == 0:
            return other
        if other.magnitude == 0:
            return self
        y_component = (math.sin(self.angle) * self.magnitude) + (math.sin(other.angle) * other.magnitude)
        x_component = (math.cos(self.angle) * self.magnitude) + (math.cos(other.angle) * other.magnitude)
        if x_component == 0:
            angle = math.pi / 2.0 if y_component >= 0 else 3 * math.pi / 2.0
        else:
            if x_component > 0:
                angle = math.atan(y_component / x_component)
            else:
                angle = math.pi + math.atan(y_component / x_component)
        magnitude = math.sqrt((y_component**2) + (x_component**2))
        return Q_Vector2D(angle, magnitude)

    @property
    def x(self):
        return math.cos(self.angle) * self.magnitude

    @property
    def y(self):
        return math.sin(self.angle) * self.magnitude

    @property
    def degrees(self):
        """
        >>> v3 = Q_Vector2D(angle=3.141592653589793, magnitude=10)
        >>> v3.x
        -10.0
        >>> v3.y
        1.2246467991473533e-15
        >>> v3.degrees
        180.0
        """
        return Q_map(self.angle, 0, math.pi * 2.0, 0, 360)

    def limit(self, maximum):
        self.magnitude = min(self.magnitude, maximum)

    @staticmethod
    def random():
        angle = random.random() * math.pi * 2.0
        magnitude = random.random()
        return Q_Vector2D(angle=angle, magnitude=magnitude)

    @staticmethod
    def fromXY(x: float = 0, y: float = 0):
        """
        >>> v4 = Q_Vector2D.fromXY(x=1, y=1)
        >>> v4
        Q_Vector2D(angle=0.7853981633974483, magnitude=1.4142135623730951)
        >>> v5 = Q_Vector2D.fromXY(x=1, y=0)
        >>> v6 = Q_Vector2D.fromXY(x=0, y=1)
        >>> v7 = v5 + v6
        >>> v7
        Q_Vector2D(angle=0.7853981633974483, magnitude=1.4142135623730951)
        """
        if x == 0:
            angle = math.pi / 2.0 if y >= 0 else 3 * math.pi / 2.0
        else:
            if x > 0:
                angle = math.atan(y / x)
            else:
                angle = math.pi + math.atan(y / x)
        magnitude = math.sqrt((y**2) + (x**2))
        return Q_Vector2D(angle, magnitude)


class Q_Vector3d:
    """
        ██████           █████   █████  ██████████    █████████   ███████████     ███████     ███████████     ████████   ██████████
      ███░░░░███        ░░███   ░░███  ░░███░░░░░█   ███░░░░░███ ░█░░░███░░░█   ███░░░░░███  ░░███░░░░░███   ███░░░░███ ░░███░░░░███
     ███    ░░███        ░███    ░███   ░███  █ ░   ███     ░░░  ░   ░███  ░   ███     ░░███  ░███    ░███  ░░░    ░███  ░███   ░░███
    ░███     ░███        ░███    ░███   ░██████    ░███              ░███     ░███      ░███  ░██████████      ██████░   ░███    ░███
    ░███   ██░███        ░░███   ███    ░███░░█    ░███              ░███     ░███      ░███  ░███░░░░░███    ░░░░░░███  ░███    ░███
    ░░███ ░░████          ░░░█████░     ░███ ░   █ ░░███     ███     ░███     ░░███     ███   ░███    ░███   ███   ░███  ░███    ███
     ░░░██████░██           ░░███       ██████████  ░░█████████      █████     ░░░███████░    █████   █████ ░░████████   ██████████
       ░░░░░░ ░░             ░░░       ░░░░░░░░░░    ░░░░░░░░░      ░░░░░        ░░░░░░░     ░░░░░   ░░░░░   ░░░░░░░░   ░░░░░░░░░░
    """

    COINCIDENT = 0.9999
    EPSILON = 0.000000001

    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    @staticmethod
    def NORM_XAXIS():
        """
        >>> Q_Vector3d.NORM_XAXIS() == Q_Vector3d(1, 0, 0)
        True
        """
        return Q_Vector3d(1, 0, 0)

    @staticmethod
    def NORM_YAXIS():
        """
        >>> Q_Vector3d.NORM_YAXIS() == Q_Vector3d(0, 1, 0)
        True
        """
        return Q_Vector3d(0, 1, 0)

    @staticmethod
    def NORM_ZAXIS():
        """
        >>> Q_Vector3d.NORM_ZAXIS() == Q_Vector3d(0, 0, 1)
        True
        """
        return Q_Vector3d(0, 0, 1)

    @staticmethod
    def from_Vector3D(other_vector):
        """
        >>> input_vector = Q_Vector3d(2.5, -0.9, 4.62)
        >>> output_vector = Q_Vector3d.from_Vector3D(input_vector)
        >>> input_vector == output_vector
        True
        """
        return Q_Vector3d(x=other_vector.x, y=other_vector.y, z=other_vector.z)

    @staticmethod
    def from_normal(normalized_vector):
        assert math.fabs(normalized_vector.dot_product(other_vector=normalized_vector) - 1.0) < Q_Vector3d.EPSILON
        return normalized_vector

    @staticmethod
    def get_normalized_vector(x: float, y: float, z: float):
        return Q_Vector3d(x=x, y=y, z=z).normalized()

    @staticmethod
    def random_in_unit_disk():
        # return Q_Vector3d(random.random() * 2 - 1, random.random() * 2 - 1, 0).normalized()
        # return Q_Vector3d(random.random() * 2 - 1, random.random() * 2 - 1, 0)
        p = Q_Vector3d(random.random() * 2 - 1, random.random() * 2 - 1, 0)
        while p.length_squared >= 1:
            p = Q_Vector3d(random.random() * 2 - 1, random.random() * 2 - 1, 0)
        return p

    @staticmethod
    def random_in_unit_sphere():
        theta = 2 * math.pi * random.random()
        phi = math.acos(1 - 2 * random.random())
        x = math.sin(phi) * math.cos(theta)
        y = math.sin(phi) * math.sin(theta)
        z = math.cos(phi)
        return Q_Vector3d(x, y, z)
        # # return Q_Vector3d(random.random() * 2 - 1, random.random() * 2 - 1, random.random() * 2 - 1).normalized()
        # # return Q_Vector3d(random.random() * 2 - 1, random.random() * 2 - 1, random.random() * 2 - 1)
        # p = Q_Vector3d(random.random() * 2 - 1, random.random() * 2 - 1, random.random() * 2 - 1)
        # while p.length_squared >= 1.0:
        #     p = Q_Vector3d(random.random() * 2 - 1, random.random() * 2 - 1, random.random() * 2 - 1)
        # return p

    def clamp(self, lower_limit: float, upper_limit: float):
        return Q_Vector3d(
            x=Q_clamp(self.x, minimum_limit=lower_limit, maximum_limit=upper_limit),
            y=Q_clamp(self.y, minimum_limit=lower_limit, maximum_limit=upper_limit),
            z=Q_clamp(self.z, minimum_limit=lower_limit, maximum_limit=upper_limit),
        )

    def to_tuple(self):
        return (self.x, self.y, self.z)

    def to_list(self):
        return [self.x, self.y, self.z]

    def dot_product(self, other_vector) -> float:
        return self.x * other_vector.x + self.y * other_vector.y + self.z * other_vector.z

    def element_wise_product(self, other_vector):
        return Q_Vector3d(x=self.x * other_vector.x, y=self.y * other_vector.y, z=self.z * other_vector.z)

    def cross_product(self, other_vector):
        """
        >>> first_vector = Q_Vector3d(x=1, y=2, z=3)
        >>> second_vector = Q_Vector3d(x=4, y=5, z=6)
        >>> expected_vector = Q_Vector3d(x=-3.0, y=6.0, z=-3.0)
        >>> cross_product = first_vector.cross_product(other_vector=second_vector)
        >>> cross_product == expected_vector
        True
        """
        return Q_Vector3d(
            x=self.y * other_vector.z - self.z * other_vector.y,
            y=self.z * other_vector.x - self.x * other_vector.z,
            z=self.x * other_vector.y - self.y * other_vector.x,
        )

    @property
    def length_squared(self):
        """
        This is included to avoide a square root calculation where possible.
        """
        return self.dot_product(other_vector=self)

    @property
    def length(self):
        """
        >>> test_vector = Q_Vector3d(x=1, y=2, z=3)
        >>> 1**2 + 2**2 + 3**2
        14
        >>> math.sqrt(1**2 + 2**2 + 3**2)
        3.7416573867739413
        >>> math.sqrt(test_vector.x**2 + test_vector.y**2 + test_vector.z**2)
        3.7416573867739413
        >>> test_vector.length
        3.7416573867739413
        """
        # return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        # This should do the same thing...
        return math.sqrt(self.dot_product(other_vector=self))

    def normalized(self):
        norm = self.length
        if norm > 0:
            norm = 1 / norm
            return self * norm
        return None

    def limit(self, limit: float):
        """
        >>> test_vector = Q_Vector3d(x=5, y=4, z=3)
        >>> test_vector.limit(10)
        >>> test_vector == Q_Vector3d(5.0, 4.0, 3.0)
        True
        >>> test_vector.limit(5)
        >>> test_vector == Q_Vector3d(3.5355339059327373, 2.82842712474619, 2.1213203435596424)
        True
        """
        length = self.length
        if length > limit:
            scalar = limit / length
            self.x *= scalar
            self.y *= scalar
            self.z *= scalar

    def reflected(self, other_vector) -> float:
        """
        >>> test_vector = Q_Vector3d(x=0, y=0, z=2)
        >>> surface_vector = Q_Vector3d(x=0, y=0, z=-1).normalized()
        >>> reflected_vector = test_vector.reflected(surface_vector)
        >>> reflected_vector == Q_Vector3d(0.0, 0.0, -2.0)
        True
        """
        # Maybe we should always normalize other_vector first?
        return Q_Vector3d.from_Vector3D(self - (other_vector * 2 * self.dot_product(other_vector=other_vector)))

    #######################################
    #
    # OPERATORS
    #
    #######################################

    def __add__(self, other_vector):
        return Q_Vector3d(x=self.x + other_vector.x, y=self.y + other_vector.y, z=self.z + other_vector.z)

    def __sub__(self, other_vector):
        return Q_Vector3d(x=self.x - other_vector.x, y=self.y - other_vector.y, z=self.z - other_vector.z)

    def __mul__(self, other_object):
        """
        >>> test_vector = Q_Vector3d(x=1, y=1, z=1)
        >>> test_vector * 2.0
        Q_Vector3d(x=2.0, y=2.0, z=2.0)

        >>> test_vector = Q_Vector3d(x=5, y=4, z=3)
        >>> mult_vector = Q_Vector3d(x=2, y=3, z=4)
        >>> test_vector * mult_vector
        Q_Vector3d(x=10.0, y=12.0, z=12.0)
        """
        if type(other_object) in (float, int):
            return Q_Vector3d(x=self.x * other_object, y=self.y * other_object, z=self.z * other_object)
        elif type(other_object) == Q_Vector3d:
            return self.element_wise_product(other_vector=other_object)
        else:
            print(type(self), type(other_object))
            raise ('Unknown Type')

    def __rmul__(self, other_object):
        if type(other_object) in (float, int):
            return Q_Vector3d(x=self.x * other_object, y=self.y * other_object, z=self.z * other_object)
        elif type(other_object) == Q_Vector3d:
            return self.element_wise_product(other_vector=other_object)
        else:
            print(type(self), type(other_object))
            raise ('Unknown Type')

    def __eq__(self, other_vector):
        if type(other_vector) != Q_Vector3d:
            print(type(self), type(other_vector))
            raise TypeError('Other object must be Q_Vector3D.')
        return (self.x == other_vector.x) and (self.y == other_vector.y) and (self.z == other_vector.z)

    def __ne__(self, other_vector):
        if type(other_vector) != Q_Vector3d:
            print(type(self), type(other_vector))
            raise TypeError('Other object must be Q_Vector3D.')
        return not self == other_vector

    def __str__(self):
        return f'{{{self.x}, {self.y}, {self.z}}}'

    def __repr__(self):
        return f'Q_Vector3d(x={self.x}, y={self.y}, z={self.z})'

    def __neg__(self):
        return self * -1


class Q_MinMaxScaler:
    def __init__(self, items: List[int | float]):
        """
        >>> items = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        >>> Q_scale = Q_MinMaxScaler(items)
        >>> Q_scale.input
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        >>> Q_scale.minimum
        0.0
        >>> Q_scale.maximum
        5.0
        """
        self.input = items
        self.minimum = min(items)
        self.maximum = max(items)

    def transform(self):
        """
        >>> items = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        >>> Q_scale = Q_MinMaxScaler(items)
        >>> Q_scale.input
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        >>> Q_scale.minimum
        0.0
        >>> Q_scale.maximum
        5.0
        >>> transformed = Q_scale.transform()
        >>> transformed
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        """
        return [Q_map(x, self.minimum, self.maximum, 0, 1) for x in self.input]

    def reverse_transform(self, values: List[int | float]):
        """
        >>> items = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        >>> Q_scale = Q_MinMaxScaler(items)
        >>> Q_scale.input
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        >>> Q_scale.minimum
        0.0
        >>> Q_scale.maximum
        5.0
        >>> transformed = Q_scale.transform()
        >>> transformed
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        >>> reverted = Q_scale.reverse_transform(transformed)
        >>> reverted
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        >>> items == reverted
        True
        >>> fake_data = [pnoise2(x / 9, 1.678) for x in range(5)]
        >>> fake_data
        [0.259732723236084, 0.2692364752292633, 0.2301626205444336, 0.14375047385692596, 0.040179312229156494]
        >>> fake_reverted = Q_scale.reverse_transform(fake_data)
        >>> fake_reverted
        [1.29866361618042, 1.3461823761463165, 1.150813102722168, 0.7187523692846298, 0.20089656114578247]
        """
        return [Q_map(x, 0, 1, self.minimum, self.maximum) for x in values]


def Q_constrain(value, lowerLimit=0, upperLimit=1):
    """
    >>> Q_constrain(value=0.5, lowerLimit=0, upperLimit=1)
    0.5
    >>> Q_constrain(value=500, lowerLimit=1000, upperLimit=1500)
    1000
    >>> Q_constrain(value=-2, lowerLimit=0, upperLimit=10)
    0
    >>> Q_constrain(value=-2, lowerLimit=-10, upperLimit=-5)
    -5
    >>> Q_constrain(value=2000.5, lowerLimit=-10, upperLimit=250.362)
    250.362
    """
    if value < lowerLimit:
        return lowerLimit
    elif value > upperLimit:
        return upperLimit
    else:
        return value


def Q_map(
    value: float, lower_limit: float, upper_limit: float, scaled_lower_limit: float, scaled_upper_limit: float
) -> float:
    """Adjusts the given input value, which falls within a linear range between lower_limit and upper_limit
    to between the separate linear range of scaled_lower_limit and scaled_upper_limit.

    Example: value = 0.5, lower_limit = 0, upper_limit = 1, scaled_lower_limit = 0, scaled_upper_limit = 100 -> 50
    Example: value = 0.1, lower_limit = 0, upper_limit = 1, scaled_lower_limit = 100, scaled_upper_limit = 200 -> 110

    >>> Q_map(value = 0.5, lower_limit = 0, upper_limit = 1, scaled_lower_limit = 0, scaled_upper_limit = 100)
    50.0
    >>> Q_map(value = 0.1, lower_limit = 0, upper_limit = 1, scaled_lower_limit = 100, scaled_upper_limit = 200)
    110.0
    >>> Q_map(value=0.9, lower_limit=0, upper_limit=1, scaled_lower_limit=0, scaled_upper_limit=100)
    90.0
    """
    temp_value = value - lower_limit
    temp_scale = temp_value / (upper_limit - lower_limit)
    return ((scaled_upper_limit - scaled_lower_limit) * temp_scale) + scaled_lower_limit


def Q_quantize(
    value: float | int, max_value: float | int, num_thresholds: float | int, min_value: float | int = 0
) -> float | int:
    # return round(value * thresholds / max_value) * max_value / thresholds
    """
    >>> Q_quantize(0, 100, 4)
    0
    >>> Q_quantize(1, 100, 4)
    0
    >>> Q_quantize(20, 100, 4)
    25
    >>> Q_quantize(26, 100, 4)
    25
    >>> Q_quantize(45, 100, 4)
    50
    >>> Q_quantize(66, 100, 4)
    75
    >>> Q_quantize(75, 100, 4)
    75
    >>> Q_quantize(99, 100, 4)
    100
    >>> Q_quantize(100, 100, 4)
    100

    >>> Q_quantize(50, min_value=50, max_value=100, num_thresholds=5)
    50

    >>> Q_quantize(54, min_value=50, max_value=100, num_thresholds=5)
    50

    >>> Q_quantize(58, min_value=50, max_value=100, num_thresholds=5)
    60

    >>> Q_quantize(88, min_value=50, max_value=100, num_thresholds=5)
    90

    >>> Q_quantize(98, min_value=50, max_value=100, num_thresholds=5)
    100

    >>> Q_quantize(37.5, min_value=48.129, max_value=141.001, num_thresholds=5)
    29.554599999999994

    >>> Q_quantize(98.242, min_value=48.129, max_value=100.001, num_thresholds=5)
    100.001

    >>> Q_quantize(100, min_value=50, max_value=100, num_thresholds=5)
    100
    """
    if type(value) == float or type(num_thresholds) == float or type(min_value) == float or type(max_value) == float:
        return_type = float
    else:
        return_type = int
    return return_type(
        round((value - min_value) * num_thresholds / (max_value - min_value)) * (max_value - min_value) / num_thresholds
        + min_value
    )


def Q_what(array: list) -> list:
    """
    >>> test_array = [[112, 48, 70, 107, 102, 102, 32, 37],[142, 39, 42, 76, 118, 81, 114, 83],[113, 133, 145, 89, 48, 59, 35, 111],[66, 67, 87, 65, 85, 68, 49, 108],[91, 59, 52, 85, 97, 47, 60, 139],[92, 58, 87, 23, 145, 20, 89, 115],[112, 139, 50, 141, 39, 102, 110, 76],[91, 65, 33, 62, 101, 119, 107, 41]]
    >>> transformed_array = Q_what(array=test_array)
    >>> transformed_array
    [[187, 57, 102, 177, 167, 167, 24, 34], [248, 38, 44, 114, 199, 124, 191, 128], [189, 230, 255, 140, 57, 79, 30, 185], [93, 95, 136, 91, 132, 97, 59, 179], [144, 79, 65, 132, 157, 55, 81, 242], [146, 77, 136, 6, 255, 0, 140, 193], [187, 242, 61, 246, 38, 167, 183, 114], [144, 91, 26, 85, 165, 201, 177, 42]]
    """
    flattened_array = list(itertools.chain.from_iterable(array))
    # flattened_array = [x for row in array for x in row]
    minimum = min(flattened_array)
    maximum = max(flattened_array)
    return [[math.floor(255 * (x - minimum) / (maximum - minimum)) for x in row] for row in array]


def Q_print_list_as_table(array: list, decimalPlaces: int = None):
    for row in array:
        for item in row:
            if type(item) in (float, int) and decimalPlaces is not None:
                if decimalPlaces > 0:
                    print(round(item, decimalPlaces), end='\t')
                else:
                    print(int(round(item, decimalPlaces)), end='\t')
            else:
                print(item, end='\t')
        print()


def Q_divideArray(divide_this_array: list, by_this_array: list, decimalPlaces: int = None) -> list:
    if len(divide_this_array) != len(by_this_array):
        return None
    result = [[0 for column in range(len(divide_this_array))] for row in range(len(divide_this_array))]
    for row in range(len(divide_this_array)):
        for column in range(len(divide_this_array)):
            if (
                type(divide_this_array[row][column]) in (float, int)
                and type(by_this_array[row][column]) in (float, int)
                and decimalPlaces is not None
            ):
                if decimalPlaces > 0:
                    result[row][column] = round(
                        divide_this_array[row][column] / by_this_array[row][column], decimalPlaces
                    )
                else:
                    result[row][column] = int(
                        round(divide_this_array[row][column] / by_this_array[row][column], decimalPlaces)
                    )
            else:
                result[row][column] = divide_this_array[row][column] / by_this_array[row][column]
    return result


def Q_multiplyArray(multiply_this_array: list, by_this_array: list, decimalPlaces: int = None) -> list:
    if len(multiply_this_array) != len(by_this_array):
        return None
    result = [[0 for column in range(len(multiply_this_array))] for row in range(len(multiply_this_array))]
    for row in range(len(multiply_this_array)):
        for column in range(len(multiply_this_array)):
            if (
                type(multiply_this_array[row][column]) in (float, int)
                and type(by_this_array[row][column]) in (float, int)
                and decimalPlaces is not None
            ):
                if decimalPlaces > 0:
                    result[row][column] = round(
                        multiply_this_array[row][column] * by_this_array[row][column], decimalPlaces
                    )
                else:
                    result[row][column] = int(multiply_this_array[row][column] * by_this_array[row][column])
            else:
                result[row][column] = multiply_this_array[row][column] * by_this_array[row][column]
    return result


def Q_subtractArray(from_this_array: list, subtract_this_array: list, decimalPlaces: int = None) -> list:
    if len(from_this_array) != len(subtract_this_array):
        return None
    result = [[0 for column in range(len(from_this_array))] for row in range(len(from_this_array))]
    for row in range(len(from_this_array)):
        for column in range(len(from_this_array)):
            if (
                type(from_this_array[row][column]) in (float, int)
                and type(subtract_this_array[row][column]) in (float, int)
                and decimalPlaces is not None
            ):
                if decimalPlaces > 0:
                    result[row][column] = round(
                        from_this_array[row][column] - subtract_this_array[row][column], decimalPlaces
                    )
                else:
                    result[row][column] = int(from_this_array[row][column] - subtract_this_array[row][column])
            else:
                result[row][column] = from_this_array[row][column] - subtract_this_array[row][column]
    return result


def Q_zigZag(array: list) -> list:
    """Returns a 1D list of elements extracted from following a zigzag path through the provided array.
    Zigzag is defined as starting at row, column position (0, 0), then:
        Moving right 1 element, or down 1 element if this is the right edge of the array
        Moving diagnally down-left until the first column is reached, or the bottom row is reached
        Moving down 1 element, or right 1 element if this is the bottom row
        Moving diagnally up-right until the first row is reached, or the right edge is reached
    - repeating the above steps until all elements have been traversed

    [[1, 2],
     [3. 4]]
    >>> Q_zigZag(array=[[1, 2], [3, 4]])
    [1, 2, 3, 4]

    [[1, 2, 3, 4, 5],
     [6, 7, 8, 9, 10]]
    >>> Q_zigZag(array=[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    [1, 2, 6, 7, 3, 4, 8, 9, 5, 10]

    [[1,  2,  3,  4,  5],
     [6,  7,  8,  9,  10],
     [11, 12, 13, 14, 15]]
    >>> Q_zigZag(array=[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
    [1, 2, 6, 11, 7, 3, 4, 8, 12, 13, 9, 5, 10, 14, 15]

    """
    DIRECTIONS = {
        'UP': (-1, 0),
        'DOWN': (1, 0),
        'LEFT': (0, -1),
        'RIGHT': (0, 1),
        'UP-RIGHT': (-1, 1),
        'DOWN-LEFT': (1, -1),
    }
    ARRAY_HEIGHT = len(array)
    ARRAY_WIDTH = len(array[0])
    STARTING_POSITION = (0, 0)
    zigzag_elements = []

    def isValidPosition(row: int, column: int) -> bool:
        return 0 <= row <= ARRAY_HEIGHT - 1 and 0 <= column <= ARRAY_WIDTH - 1

    row, column = STARTING_POSITION
    direction = (0, 0)
    while row < ARRAY_HEIGHT and column < ARRAY_WIDTH:
        # print(f'Tring to append row: {row}  column: {column}')
        zigzag_elements.append(array[row][column])
        if row == 0:  # Top of array - try to move right; move down instead if this is the top-right of the array.
            if isValidPosition(row=row, column=column + 1):
                direction = DIRECTIONS['RIGHT']
                row, column = row + direction[0], column + direction[1]
                # print(f'Tring to append row: {row}  column: {column}')
                zigzag_elements.append(array[row][column])
                direction = DIRECTIONS['DOWN-LEFT']
            elif isValidPosition(row=row + 1, column=column):
                direction = DIRECTIONS['DOWN']
                row, column = row + direction[0], column + direction[1]
                # print(f'Tring to append row: {row}  column: {column}')
                zigzag_elements.append(array[row][column])
                direction = DIRECTIONS['DOWN-LEFT']
        elif row == ARRAY_HEIGHT - 1 and column < ARRAY_WIDTH - 1:  # Bottom of array, but not bottom-right corner.
            if isValidPosition(row=row, column=column + 1):
                direction = DIRECTIONS['RIGHT']
                row, column = row + direction[0], column + direction[1]
                # print(f'Tring to append row: {row}  column: {column}')
                zigzag_elements.append(array[row][column])
                direction = DIRECTIONS['UP-RIGHT']
        elif (
            column == 0
        ):  # Left of array -  try to move down; move right instead if this is the bottom-left of the array.
            if isValidPosition(row=row + 1, column=column):
                direction = DIRECTIONS['DOWN']
                row, column = row + direction[0], column + direction[1]
                # print(f'Tring to append row: {row}  column: {column}')
                zigzag_elements.append(array[row][column])
                direction = DIRECTIONS['UP-RIGHT']
            elif isValidPosition(row=row, column=column + 1):
                direction = DIRECTIONS['RIGHT']
                row, column = row + direction[0], column + direction[1]
                # print(f'Tring to append row: {row}  column: {column}')
                zigzag_elements.append(array[row][column])
                direction = DIRECTIONS['UP-RIGHT']
        elif column == ARRAY_WIDTH - 1 and row > 0:  # Right-edge of array, but not top-right corner.
            if isValidPosition(row=row + 1, column=column):
                direction = DIRECTIONS['DOWN']
                row, column = row + direction[0], column + direction[1]
                # print(f'Tring to append row: {row}  column: {column}')
                zigzag_elements.append(array[row][column])
                direction = DIRECTIONS['DOWN-LEFT']
        row, column = row + direction[0], column + direction[1]
    return zigzag_elements


def Q_DCT(array: list, dct_type: str = 'II') -> list:
    assert dct_type == 'II'
    ARRAY_DIM = 8
    dct = [[0 for column in range(ARRAY_DIM)] for row in range(ARRAY_DIM)]
    sqrt_2 = math.sqrt(2)
    sqrt_8 = math.sqrt(ARRAY_DIM)
    for y, row in enumerate(array):
        for x, column in enumerate(row):
            if y == 0:
                ci = 1 / sqrt_8
            else:
                ci = sqrt_2 / sqrt_8

            if x == 0:
                cj = 1 / sqrt_8
            else:
                cj = sqrt_2 / sqrt_8

            sum_value = 0
            for yy in range(ARRAY_DIM):
                for xx in range(ARRAY_DIM):
                    pi = math.pi  # 3.142857
                    dct1 = (
                        array[yy][xx]
                        * math.cos((2 * yy + 1) * y * pi / (2 * ARRAY_DIM))
                        * math.cos((2 * xx + 1) * x * pi / (2 * ARRAY_DIM))
                    )
                    sum_value += dct1

            dct[y][x] = round(ci * cj * sum_value, ndigits=4)
    return dct


def Q_IDCT(array: list, dct_type: str = 'III'):
    assert dct_type == 'III'
    ARRAY_DIM = 8
    idct = [[0 for column in range(ARRAY_DIM)] for row in range(ARRAY_DIM)]
    sqrt_2 = math.sqrt(2)
    sqrt_8 = math.sqrt(ARRAY_DIM)
    for x, row in enumerate(array):
        for y, column in enumerate(row):
            sum_value = 0
            for u in range(ARRAY_DIM):
                for v in range(ARRAY_DIM):
                    if u == 0:
                        ci = 1 / sqrt_8
                    else:
                        ci = sqrt_2 / sqrt_8

                    if v == 0:
                        cj = 1 / sqrt_8
                    else:
                        cj = sqrt_2 / sqrt_8
                    pi = math.pi  # 3.142857
                    # dct1 = ci * cj * array[yy][xx] * math.cos((2 * yy + 1) * y * pi / (2 * ARRAY_DIM)) * math.cos((2 * xx + 1) * x * pi / (2 * ARRAY_DIM))
                    dct1 = (
                        ci
                        * cj
                        * array[u][v]
                        * math.cos((2 * y + 1) * v * pi / (2 * ARRAY_DIM))
                        * math.cos((2 * x + 1) * u * pi / (2 * ARRAY_DIM))
                    )
                    sum_value += dct1

            idct[x][y] = round(sum_value, ndigits=4)
    return idct


def test_computerphile_idct():
    # # Q_DCT
    # print('Q_DCT')
    # # test_array = [[Q_quantize((1 + x) * (1 + y), 64, 5) for x in range(8)] for y in range(8)]
    # test_array = [[45, 18, 47, 41, 14, 11, 37, 32],
    #               [13, 11, 43, 12, 26, 8, 10, 15],
    #               [20, 19, 31, 39, 17, 12, 34, 47],
    #               [27, 15, 28, 33, 5, 17, 27, 35],
    #               [45, 34, 26, 19, 1, 49, 39, 21],
    #               [13, 7, 1, 46, 4, 21, 22, 17],
    #               [40, 8, 12, 41, 40, 28, 38, 13],
    #               [47, 43, 5, 26, 1, 2, 6, 11]]
    # Q_printList(test_array)
    # transformed_array = Q_DCT(array=test_array)
    # Q_printList(transformed_array)
    # print('---------------------------')
    # Q_printList(test_array)
    # Q_printList(Q_IDCT(array=transformed_array))

    # print(scipy.fftpack.dct(test_array, type=4))
    # Computerphile based test
    # Source: https://www.youtube.com/watch?v=Q2aEzeMDHMA
    # print('--------Computerphile tests-------------------')
    test_array = [
        [62, 55, 55, 54, 49, 48, 47, 55],
        [62, 57, 54, 52, 48, 47, 48, 53],
        [61, 60, 52, 49, 48, 47, 49, 54],
        [63, 61, 60, 60, 63, 65, 68, 65],
        [67, 67, 70, 74, 79, 85, 91, 92],
        [82, 95, 101, 106, 114, 115, 112, 117],
        [96, 111, 115, 119, 128, 128, 130, 127],
        [109, 121, 127, 133, 139, 141, 140, 133],
    ]
    # print('--------Original array-------------------')
    # Q_print_list_as_table(test_array)
    shifted_test_array = [[x - 128 for x in row] for row in test_array]
    # print('--------Shifted -128 array-------------------')
    # Q_print_list_as_table(shifted_test_array)
    # print(shifted_test_array)
    expected_shifted_test_array = [
        [-66, -73, -73, -74, -79, -80, -81, -73],
        [-66, -71, -74, -76, -80, -81, -80, -75],
        [-67, -68, -76, -79, -80, -81, -79, -74],
        [-65, -67, -68, -68, -65, -63, -60, -63],
        [-61, -61, -58, -54, -49, -43, -37, -36],
        [-46, -33, -27, -22, -14, -13, -16, -11],
        [-32, -17, -13, -9, 0, 0, 2, -1],
        [-19, -7, -1, 5, 11, 13, 12, 5],
    ]
    assert shifted_test_array == expected_shifted_test_array
    # print('--------DCT2 Coefficients-------------------')
    transformed_array = Q_DCT(array=shifted_test_array)
    expected_transformed_array = [
        [-369.625, -29.672, -2.6411, -2.4719, -1.125, -3.711, -1.4767, -0.0779],
        [-231.0754, 44.9223, 24.4854, -0.2736, 9.2988, 3.913, 4.2906, -1.3505],
        [62.8467, 8.5314, -7.581, -2.6598, 0.315, -0.408, 0.5063, -0.8294],
        [12.4972, -14.6065, -3.4845, -3.4424, 2.4257, -1.3262, 2.7164, -0.383],
        [-4.875, -3.8564, 0.8726, 3.5645, 0.125, 5.1243, 1.1268, 0.4765],
        [-0.4752, 3.1936, -1.4333, 0.2042, -1.0595, -1.4831, -1.1313, 0.904],
        [4.4103, 2.2848, -1.7437, -1.566, 1.0872, -2.741, 1.081, -1.4058],
        [-10.1881, -1.8202, 5.9051, -0.4234, 0.2984, 0.4157, -0.9783, 0.0032],
    ]
    assert transformed_array == expected_transformed_array
    # print(transformed_array)
    # Q_print_list_as_table(transformed_array, decimalPlaces=1)
    # print('----------Quantized Output-----------------')
    quantization_table = [
        [16, 12, 14, 14, 18, 24, 49, 72],
        [11, 12, 13, 17, 22, 35, 64, 92],
        [10, 14, 16, 22, 37, 55, 78, 95],
        [16, 19, 24, 29, 56, 64, 87, 98],
        [24, 26, 40, 51, 68, 81, 103, 112],
        [40, 58, 57, 87, 109, 104, 121, 100],
        [51, 60, 69, 80, 103, 113, 120, 103],
        [61, 55, 56, 62, 77, 92, 101, 99],
    ]
    quantized_array = Q_divideArray(
        divide_this_array=transformed_array, by_this_array=quantization_table, decimalPlaces=0
    )
    # print(quantized_array)
    expected_quantized_array = [
        [-23, -2, 0, 0, 0, 0, 0, 0],
        [-21, 4, 2, 0, 0, 0, 0, 0],
        [6, 1, 0, 0, 0, 0, 0, 0],
        [1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    assert quantized_array == expected_quantized_array

    # Q_print_list_as_table(array=quantized_array)
    # print('----------DCT3 Coefficients-----------------')
    dct3_coefficients = Q_multiplyArray(multiply_this_array=quantized_array, by_this_array=quantization_table)
    expected_dct3_coefficients = [
        [-368, -24, 0, 0, 0, 0, 0, 0],
        [-231, 48, 26, 0, 0, 0, 0, 0],
        [60, 14, 0, 0, 0, 0, 0, 0],
        [16, -19, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    assert dct3_coefficients == expected_dct3_coefficients
    # Q_print_list_as_table(array=dct3_coefficients)
    # print('----------Output Shifted Block-----------------')
    output_shifted_block = Q_IDCT(array=dct3_coefficients)
    expected_output_shifted_block = [
        [-61.33, -65.7971, -72.5555, -78.4609, -81.1184, -80.1234, -77.1232, -74.69],
        [-63.6058, -67.7254, -74.0693, -79.8785, -83.0003, -82.9596, -81.0306, -79.3004],
        [-66.5505, -69.3625, -73.711, -77.7358, -79.9768, -80.0929, -78.9136, -77.8169],
        [-66.6139, -66.9274, -67.2093, -66.9957, -66.0218, -64.4357, -62.7764, -61.7174],
        [-59.6497, -57.1035, -52.6963, -47.5198, -42.6596, -38.8556, -36.3895, -35.2159],
        [-44.517, -40.0381, -32.6094, -24.5601, -17.9631, -13.8226, -11.9217, -11.3515],
        [-25.9712, -20.9846, -13.0387, -5.1364, 0.2511, 2.3035, 1.9765, 1.1133],
        [-13.0509, -8.2824, -0.9672, 5.6656, 9.1105, 8.843, 6.3996, 4.2678],
    ]
    assert output_shifted_block == expected_output_shifted_block
    # Q_print_list_as_table(array=output_shifted_block, decimalPlaces=0)
    # print('----------Output Block-----------------')
    output_block = [[x + 128 for x in row] for row in output_shifted_block]
    # Q_print_list_as_table(array=output_block, decimalPlaces=0)
    expecte_output_block = [
        [
            66.67,
            62.2029,
            55.444500000000005,
            49.539100000000005,
            46.881600000000006,
            47.876599999999996,
            50.8768,
            53.31,
        ],
        [
            64.3942,
            60.27460000000001,
            53.9307,
            48.1215,
            44.999700000000004,
            45.040400000000005,
            46.96939999999999,
            48.699600000000004,
        ],
        [61.4495, 58.6375, 54.289, 50.2642, 48.0232, 47.9071, 49.0864, 50.183099999999996],
        [61.3861, 61.072599999999994, 60.7907, 61.0043, 61.9782, 63.5643, 65.2236, 66.2826],
        [68.3503, 70.8965, 75.30369999999999, 80.4802, 85.3404, 89.14439999999999, 91.6105, 92.7841],
        [83.483, 87.9619, 95.3906, 103.4399, 110.0369, 114.1774, 116.0783, 116.6485],
        [102.0288, 107.0154, 114.9613, 122.8636, 128.2511, 130.3035, 129.9765, 129.1133],
        [114.9491, 119.7176, 127.0328, 133.6656, 137.1105, 136.843, 134.3996, 132.2678],
    ]
    assert output_block == expecte_output_block
    # print('--------Original array-------------------')
    # Q_print_list_as_table(test_array)
    # print('--------Difference-------------------')
    diff_array = Q_subtractArray(from_this_array=test_array, subtract_this_array=output_block, decimalPlaces=0)
    expected_diff_array = [
        [-4, -7, 0, 4, 2, 0, -3, 1],
        [-2, -3, 0, 3, 3, 1, 1, 4],
        [0, 1, -2, -1, 0, 0, 0, 3],
        [1, 0, 0, -1, 1, 1, 2, -1],
        [-1, -3, -5, -6, -6, -4, 0, 0],
        [-1, 7, 5, 2, 3, 0, -4, 0],
        [-6, 3, 0, -3, 0, -2, 0, -2],
        [-5, 1, 0, 0, 1, 4, 5, 0],
    ]
    assert diff_array == expected_diff_array


def main():
    print(Q_generate_prime())
    pass


#     banner = """
#  █████       ██████████ █████ █████      █████████     ███████    ██████   ██████ ███████████  █████ ██████   █████   █████████   ███████████ █████    ███████    ██████   █████  █████████
# ░░███       ░░███░░░░░█░░███ ░░███      ███░░░░░███  ███░░░░░███ ░░██████ ██████ ░░███░░░░░███░░███ ░░██████ ░░███   ███░░░░░███ ░█░░░███░░░█░░███   ███░░░░░███ ░░██████ ░░███  ███░░░░░███
#  ░███        ░███  █ ░  ░░███ ███      ███     ░░░  ███     ░░███ ░███░█████░███  ░███    ░███ ░███  ░███░███ ░███  ░███    ░███ ░   ░███  ░  ░███  ███     ░░███ ░███░███ ░███ ░███    ░░░
#  ░███        ░██████     ░░█████      ░███         ░███      ░███ ░███░░███ ░███  ░██████████  ░███  ░███░░███░███  ░███████████     ░███     ░███ ░███      ░███ ░███░░███░███ ░░█████████
#  ░███        ░███░░█      ███░███     ░███         ░███      ░███ ░███ ░░░  ░███  ░███░░░░░███ ░███  ░███ ░░██████  ░███░░░░░███     ░███     ░███ ░███      ░███ ░███ ░░██████  ░░░░░░░░███
#  ░███      █ ░███ ░   █  ███ ░░███    ░░███     ███░░███     ███  ░███      ░███  ░███    ░███ ░███  ░███  ░░█████  ░███    ░███     ░███     ░███ ░░███     ███  ░███  ░░█████  ███    ░███
#  ███████████ ██████████ █████ █████    ░░█████████  ░░░███████░   █████     █████ ███████████  █████ █████  ░░█████ █████   █████    █████    █████ ░░░███████░   █████  ░░█████░░█████████
# ░░░░░░░░░░░ ░░░░░░░░░░ ░░░░░ ░░░░░      ░░░░░░░░░     ░░░░░░░    ░░░░░     ░░░░░ ░░░░░░░░░░░  ░░░░░ ░░░░░    ░░░░░ ░░░░░   ░░░░░    ░░░░░    ░░░░░    ░░░░░░░    ░░░░░    ░░░░░  ░░░░░░░░░
# """
#     print(banner)
#     for i in range(6):
#         combinations.clear()
#         points = [x + 1 for x in range(i + 1)]
#         start_time = datetime.datetime.now()
#         Q_get_lex_combinations(array=sorted(points), number_of_items=len(points))
#         # print(combinations)
#         print(points, f'{sum(1 for _ in combinations):0,}', f'combinations in {datetime.datetime.now() - start_time}.')

#     combinations.clear()
#     # https://github.com/OmarAflak/RayTracer-CPP/blob/master/main.cpp


if __name__ == '__main__':
    main()
    doctest.testmod(verbose=False)
