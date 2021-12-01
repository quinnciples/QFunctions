from noise import pnoise2
import math
import itertools
import random
import datetime

global combinations
combinations = []


def Q_get_combinations(array: list, number_of_items: int, selection: list = []):
    if number_of_items <= 0:
        combinations.append([_ for _ in selection])
        print(f'selection {selection}')
        return

    for idx in range(len(array) - number_of_items + 1):
        new_array = array[idx + 1:]
        Q_get_combinations(array=new_array, number_of_items=number_of_items - 1, selection=selection + [array[idx]])


def Q_get_lex_combinations(array: list, number_of_items: int, selection: list = []) -> list:
    if number_of_items == 0:
        combinations.append(selection)
        return

    for idx in range(len(array)):
        Q_get_lex_combinations(array=array[:idx] + array[idx + 1:], number_of_items=number_of_items - 1, selection=selection + [array[idx]])


def Q_get_lex_combinations_generator(array: list, number_of_items: int, selection: list = []) -> list:
    if number_of_items == 0:
        yield selection

    for idx in range(len(array)):
        yield from Q_get_lex_combinations_generator(array=array[:idx] + array[idx + 1:], number_of_items=number_of_items - 1, selection=selection + [array[idx]])


def Q_weighted_choice(list_of_choices: list, number_of_choices: int = 1, replacement: bool = False):
    all_choices = [_ for _ in list_of_choices]
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
    all_choices = [_ for _ in list_of_choices]
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


def Q_weighted_choice3(list_of_choices: list, list_of_weights: list, number_of_choices: int = 1, replacement: bool = False):
    all_choices = [_ for _ in list_of_choices]
    all_weights = [_ for _ in list_of_weights]
    assert len(all_choices) == len(all_weights)
    results = []
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
        magnitude = math.sqrt((y_component ** 2) + (x_component ** 2))
        return Q_Vector2D(angle, magnitude)

    @property
    def x(self):
        return math.cos(self.angle) * self.magnitude

    @property
    def y(self):
        return math.sin(self.angle) * self.magnitude

    @property
    def degrees(self):
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
        if x == 0:
            angle = math.pi / 2.0 if y >= 0 else 3 * math.pi / 2.0
        else:
            if x > 0:
                angle = math.atan(y / x)
            else:
                angle = math.pi + math.atan(y / x)
        magnitude = math.sqrt((y ** 2) + (x ** 2))
        return Q_Vector2D(angle, magnitude)


class Q_MinMaxScaler:
    def __init__(self, items):
        self.input = items
        self.minimum = min(items)
        self.maximum = max(items)

    def transform(self):
        output = [Q_map(x, self.minimum, self.maximum, 0, 1) for x in self.input]
        return output

    def reverse_transform(self, values):
        output = [Q_map(x, 0, 1, self.minimum, self.maximum) for x in values]
        return output


def Q_constrain(value, lowerLimit=0, upperLimit=1):
    if value < lowerLimit:
        return lowerLimit
    elif value > upperLimit:
        return upperLimit
    else:
        return value


def Q_map(value: float, lower_limit: float, upper_limit: float, scaled_lower_limit: float, scaled_upper_limit: float) -> float:
    """Adjusts the given input value, which falls within a linear range between lower_limit and upper_limit
    to between the separate linear range of scaled_lower_limit and scaled_upper_limit.

    Example: value = 0.5, lower_limit = 0, upper_limit = 1, scaled_lower_limit = 0, scaled_upper_limit = 100 -> 50
    Example: value = 0.1, lower_limit = 0, upper_limit = 1, scaled_lower_limit = 100, scaled_upper_limit = 200 -> 110
    """
    temp_value = value - lower_limit
    temp_scale = temp_value / (upper_limit - lower_limit)
    output = ((scaled_upper_limit - scaled_lower_limit) * temp_scale) + scaled_lower_limit
    return output


def Q_quantize(value: float | int, max_value: float | int, num_thresholds: float | int, min_value: float | int = 0) -> float | int:
    # return round(value * thresholds / max_value) * max_value / thresholds
    if type(value) == float or type(num_thresholds) == float or type(min_value) == float or type(max_value) == float:
        return_type = float
    else:
        return_type = int
    return return_type(round((value - min_value) * num_thresholds / (max_value - min_value)) * (max_value - min_value) / num_thresholds + min_value)


def Q_what(array: list) -> list:
    flattened_array = list(itertools.chain.from_iterable(array))
    # flattened_array = [x for row in array for x in row]
    minimum = min(flattened_array)
    maximum = max(flattened_array)
    new_array = [[math.floor(255 * (x - minimum) / (maximum - minimum)) for x in row] for row in array]
    return new_array


def Q_printList(array: list, decimalPlaces: int = None):
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
            if type(divide_this_array[row][column]) in (float, int) and type(by_this_array[row][column]) in (float, int) and decimalPlaces is not None:
                if decimalPlaces > 0:
                    result[row][column] = round(divide_this_array[row][column] / by_this_array[row][column], decimalPlaces)
                else:
                    result[row][column] = int(round(divide_this_array[row][column] / by_this_array[row][column], decimalPlaces))
            else:
                result[row][column] = divide_this_array[row][column] / by_this_array[row][column]
    return result


def Q_multiplyArray(multiply_this_array: list, by_this_array: list, decimalPlaces: int = None) -> list:
    if len(multiply_this_array) != len(by_this_array):
        return None
    result = [[0 for column in range(len(multiply_this_array))] for row in range(len(multiply_this_array))]
    for row in range(len(multiply_this_array)):
        for column in range(len(multiply_this_array)):
            if type(multiply_this_array[row][column]) in (float, int) and type(by_this_array[row][column]) in (float, int) and decimalPlaces is not None:
                if decimalPlaces > 0:
                    result[row][column] = round(multiply_this_array[row][column] * by_this_array[row][column], decimalPlaces)
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
            if type(from_this_array[row][column]) in (float, int) and type(subtract_this_array[row][column]) in (float, int) and decimalPlaces is not None:
                if decimalPlaces > 0:
                    result[row][column] = round(from_this_array[row][column] - subtract_this_array[row][column], decimalPlaces)
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
    """
    DIRECTIONS = {'UP': (-1, 0),
                  'DOWN': (1, 0),
                  'LEFT': (0, -1),
                  'RIGHT': (0, 1),
                  'UP-RIGHT': (-1, 1),
                  'DOWN-LEFT': (1, -1)
                  }
    ARRAY_HEIGHT = len(array)
    ARRAY_WIDTH = len(array[0])
    STARTING_POSITION = (0, 0)
    zigzag_elements = []

    def isValidPosition(row: int, column: int) -> bool:
        if 0 <= row <= ARRAY_HEIGHT - 1:
            if 0 <= column <= ARRAY_WIDTH - 1:
                # print(f'Testing row: {row}  column {column} - PASS')
                return True
        # print(f'Testing row: {row}  column {column} - FAIL')
        return False

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
        elif column == 0:  # Left of array -  try to move down; move right instead if this is the bottom-left of the array.
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
    assert(dct_type == 'II')
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
                    dct1 = array[yy][xx] * math.cos((2 * yy + 1) * y * pi / (2 * ARRAY_DIM)) * math.cos((2 * xx + 1) * x * pi / (2 * ARRAY_DIM))
                    sum_value += dct1

            dct[y][x] = round(ci * cj * sum_value, ndigits=4)
    return dct


def Q_IDCT(array: list, dct_type: str = 'III'):
    assert(dct_type == 'III')
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
                    dct1 = ci * cj * array[u][v] * math.cos((2 * y + 1) * v * pi / (2 * ARRAY_DIM)) * math.cos((2 * x + 1) * u * pi / (2 * ARRAY_DIM))
                    sum_value += dct1

            idct[x][y] = round(sum_value, ndigits=4)
    return idct


def main():
    # Q_constrain tests
    print('Q_constrain')
    print(Q_constrain(value=0.5, lowerLimit=0, upperLimit=1))
    assert(Q_constrain(value=0.5, lowerLimit=0, upperLimit=1) == 0.5)
    print(Q_constrain(value=500, lowerLimit=1000, upperLimit=1500))
    assert(Q_constrain(value=500, lowerLimit=1000, upperLimit=1500) == 1000)
    print(Q_constrain(value=-2, lowerLimit=0, upperLimit=10))
    assert(Q_constrain(value=-2, lowerLimit=0, upperLimit=10) == 0)
    print(Q_constrain(value=-2, lowerLimit=-10, upperLimit=-5))
    assert(Q_constrain(value=-2, lowerLimit=-10, upperLimit=-5) == -5)
    print(Q_constrain(value=2000.5, lowerLimit=-10, upperLimit=250.362))
    assert(Q_constrain(value=2000.5, lowerLimit=-10, upperLimit=250.362) == 250.362)

    # Q_map tests
    print('Q_map')
    print(Q_map(value=0.5, lower_limit=0, upper_limit=1, scaled_lower_limit=0, scaled_upper_limit=100))
    assert(Q_map(value=0.5, lower_limit=0, upper_limit=1, scaled_lower_limit=0, scaled_upper_limit=100) == 50)
    print(Q_map(value=0.9, lower_limit=0, upper_limit=1, scaled_lower_limit=0, scaled_upper_limit=100))
    assert(Q_map(value=0.9, lower_limit=0, upper_limit=1, scaled_lower_limit=0, scaled_upper_limit=100) == 90)

    # Q_MinMaxScaler tests
    print('Q_MinMaxScaler')
    items = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    print(items)
    Q_scale = Q_MinMaxScaler(items)
    transformed = Q_scale.transform()
    print(transformed)
    reverted = Q_scale.reverse_transform(transformed)
    print(reverted)
    assert(items == reverted)
    fake_data = [pnoise2(x / 9, 1.678) for x in range(5)]
    print(fake_data)
    fake_reverted = Q_scale.reverse_transform(fake_data)
    print(fake_reverted)

    # Q_quantize tests
    print('Q_quantize')
    print(0, Q_quantize(0, 100, 4))
    assert(Q_quantize(0, 100, 4) == 0)
    print(1, Q_quantize(1, 100, 4))
    assert(Q_quantize(1, 100, 4) == 0)
    print(20, Q_quantize(20, 100, 4))
    assert(Q_quantize(20, 100, 4) == 25)
    print(26, Q_quantize(26, 100, 4))
    assert(Q_quantize(26, 100, 4) == 25)
    print(45, Q_quantize(45, 100, 4))
    assert(Q_quantize(45, 100, 4) == 50)
    print(66, Q_quantize(66, 100, 4))
    assert(Q_quantize(66, 100, 4) == 75)
    print(75, Q_quantize(75, 100, 4))
    assert(Q_quantize(75, 100, 4) == 75)
    print(99, Q_quantize(99, 100, 4))
    assert(Q_quantize(99, 100, 4) == 100)
    print(100, Q_quantize(100, 100, 4))
    assert(Q_quantize(100, 100, 4) == 100)

    print(50, Q_quantize(50, min_value=50, max_value=100, num_thresholds=5))
    assert(Q_quantize(50, min_value=50, max_value=100, num_thresholds=5) == 50)
    print(54, Q_quantize(54, min_value=50, max_value=100, num_thresholds=5))
    assert(Q_quantize(54, min_value=50, max_value=100, num_thresholds=5) == 50)
    print(58, Q_quantize(58, min_value=50, max_value=100, num_thresholds=5))
    assert(Q_quantize(58, min_value=50, max_value=100, num_thresholds=5) == 60)
    print(88, Q_quantize(88, min_value=50, max_value=100, num_thresholds=5))
    assert(Q_quantize(88, min_value=50, max_value=100, num_thresholds=5) == 90)
    print(98, Q_quantize(98, min_value=50, max_value=100, num_thresholds=5))
    assert(Q_quantize(98, min_value=50, max_value=100, num_thresholds=5) == 100)
    print(100, Q_quantize(100, min_value=50, max_value=100, num_thresholds=5))
    assert(Q_quantize(100, min_value=50, max_value=100, num_thresholds=5) == 100)

    # Q_what
    print('Q_what')
    test_array = [[112, 48, 70, 107, 102, 102, 32, 37], [142, 39, 42, 76, 118, 81, 114, 83], [113, 133, 145, 89, 48, 59, 35, 111], [66, 67, 87, 65, 85, 68, 49, 108], [91, 59, 52, 85, 97, 47, 60, 139], [92, 58, 87, 23, 145, 20, 89, 115], [112, 139, 50, 141, 39, 102, 110, 76], [91, 65, 33, 62, 101, 119, 107, 41]]
    # test_array = [[random.randint(20, 147) for columns in range(8)] for rows in range(8)]
    print(test_array)
    transformed_array = Q_what(array=test_array)
    print(transformed_array)
    assert(transformed_array == [[187, 57, 102, 177, 167, 167, 24, 34], [248, 38, 44, 114, 199, 124, 191, 128], [189, 230, 255, 140, 57, 79, 30, 185], [93, 95, 136, 91, 132, 97, 59, 179], [144, 79, 65, 132, 157, 55, 81, 242], [146, 77, 136, 6, 255, 0, 140, 193], [187, 242, 61, 246, 38, 167, 183, 114], [144, 91, 26, 85, 165, 201, 177, 42]])

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
    print('--------Computerphile tests-------------------')
    test_array = [[62, 55, 55, 54, 49, 48, 47, 55],
                  [62, 57, 54, 52, 48, 47, 48, 53],
                  [61, 60, 52, 49, 48, 47, 49, 54],
                  [63, 61, 60, 60, 63, 65, 68, 65],
                  [67, 67, 70, 74, 79, 85, 91, 92],
                  [82, 95, 101, 106, 114, 115, 112, 117],
                  [96, 111, 115, 119, 128, 128, 130, 127],
                  [109, 121, 127, 133, 139, 141, 140, 133]]
    print('--------Original array-------------------')
    Q_printList(test_array)
    shifted_test_array = [[x - 128 for x in row] for row in test_array]
    print('--------Shifted -128 array-------------------')
    Q_printList(shifted_test_array)
    print('--------DCT2 Coefficients-------------------')
    transformed_array = Q_DCT(array=shifted_test_array)
    Q_printList(transformed_array, decimalPlaces=1)
    quantization_table = [[16, 12, 14, 14, 18, 24, 49, 72],
                          [11, 12, 13, 17, 22, 35, 64, 92],
                          [10, 14, 16, 22, 37, 55, 78, 95],
                          [16, 19, 24, 29, 56, 64, 87, 98],
                          [24, 26, 40, 51, 68, 81, 103, 112],
                          [40, 58, 57, 87, 109, 104, 121, 100],
                          [51, 60, 69, 80, 103, 113, 120, 103],
                          [61, 55, 56, 62, 77, 92, 101, 99]]
    quantized_array = Q_divideArray(divide_this_array=transformed_array, by_this_array=quantization_table, decimalPlaces=0)
    print('----------Quantized Output-----------------')
    Q_printList(array=quantized_array)
    print('----------DCT3 Coefficients-----------------')
    dct3_coefficients = Q_multiplyArray(multiply_this_array=quantized_array, by_this_array=quantization_table)
    Q_printList(array=dct3_coefficients)
    print('----------Output Shifted Block-----------------')
    output_shifted_block = Q_IDCT(array=dct3_coefficients)
    Q_printList(array=output_shifted_block, decimalPlaces=0)
    print('----------Output Block-----------------')
    output_block = [[x + 128 for x in row] for row in output_shifted_block]
    Q_printList(array=output_block, decimalPlaces=0)
    print('--------Original array-------------------')
    Q_printList(test_array)
    print('--------Difference-------------------')
    Q_printList(Q_subtractArray(from_this_array=test_array, subtract_this_array=output_block, decimalPlaces=0))
    print('--------ZIG ZAG-------------------')
    zigzag_test = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    zigzag_output = Q_zigZag(array=zigzag_test)
    Q_printList(array=zigzag_test)
    print(zigzag_output)
    print('--------ZIG ZAG 2-------------------')
    zigzag_test = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    zigzag_output = Q_zigZag(array=zigzag_test)
    Q_printList(array=zigzag_test)
    print(zigzag_output)
    print('--------ZIG ZAG 3-------------------')
    zigzag_test = [[1, 2], [3, 4]]
    zigzag_output = Q_zigZag(array=zigzag_test)
    Q_printList(array=zigzag_test)
    print(zigzag_output)
    PI = math.pi
    TWO_PI = math.pi * 2.0
    v1 = Q_Vector2D(angle=PI, magnitude=10)
    v2 = Q_Vector2D(angle=PI, magnitude=0)
    print(v1, v2)
    v3 = v1 + v2
    print(v3, v3.x, v3.y, v3.degrees)
    v4 = Q_Vector2D.fromXY(x=1, y=1)
    print(v4)
    v5 = Q_Vector2D.fromXY(x=1, y=0)
    v6 = Q_Vector2D.fromXY(x=0, y=1)
    print(v5, v6, '+', v5 + v6)

#     possibilities = [(0, .35), (1, .15), (2, 0.25), (3, 0.25)]
#     results = {}
#     test_iterations = 500_000
#     start_time = datetime.datetime.now()
#     for _ in range(test_iterations):
#         i = Q_weighted_choice(possibilities, number_of_choices=4, replacement=True)
#         # print(i)
#         results[i[0]] = results.get(i[0], 0) + 1
#         results[i[1]] = results.get(i[1], 0) + 1
#         results[i[2]] = results.get(i[2], 0) + 1
#         results[i[3]] = results.get(i[3], 0) + 1
#     for k in sorted(results):
#         print(k, possibilities[k][1], results[k] / (test_iterations * 4.0))
#     print(datetime.datetime.now() - start_time)

#     possibilities = [(0, .35), (1, .15), (2, 0.25), (3, 0.25)]
#     results = {}
#     test_iterations = 50_000
#     start_time = datetime.datetime.now()
#     for _ in range(test_iterations):
#         i = Q_weighted_choice2(possibilities, number_of_choices=4, replacement=True)
#         # print(i)
#         results[i[0]] = results.get(i[0], 0) + 1
#         results[i[1]] = results.get(i[1], 0) + 1
#         results[i[2]] = results.get(i[2], 0) + 1
#         results[i[3]] = results.get(i[3], 0) + 1
#     for k in sorted(results):
#         print(k, possibilities[k][1], results[k] / (test_iterations * 4.0))
#     print(datetime.datetime.now() - start_time)

#     possibilities = [0, 1, 2, 3]
#     weights = [0.35, 0.15, 0.25, 0.25]
#     results = {}
#     test_iterations = 50_000
#     start_time = datetime.datetime.now()
#     for _ in range(test_iterations):
#         i = Q_weighted_choice3(list_of_choices=possibilities, list_of_weights=weights, number_of_choices=4, replacement=True)
#         results[i[0]] = results.get(i[0], 0) + 1
#         results[i[1]] = results.get(i[1], 0) + 1
#         results[i[2]] = results.get(i[2], 0) + 1
#         results[i[3]] = results.get(i[3], 0) + 1
#     for k in sorted(results):
#         print(k, weights[k], results[k] / (test_iterations * 4.0))
#     print(datetime.datetime.now() - start_time)

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
#     for i in range(6):
#         points = [x + 1 for x in range(i + 1)]
#         start_time = datetime.datetime.now()
#         print(points, f'{sum(1 for _ in Q_get_lex_combinations_generator(array=sorted(points), number_of_items=len(points))):0,}', f'combinations in {datetime.datetime.now() - start_time}.')


if __name__ == "__main__":
    main()
