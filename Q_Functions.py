from noise import pnoise2


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


def Q_map(value, lowerLimit, upperLimit, scaledLowerLimit, scaledUpperLimit):
    temp_val = value - lowerLimit
    temp_scale = temp_val / (upperLimit - lowerLimit)
    temp_output = ((scaledUpperLimit - scaledLowerLimit) * temp_scale) + scaledLowerLimit
    return temp_output


def Q_quantize(value: (float, int), max_value: (float, int), thresholds: (float, int), min_value: (float, int) = 0) -> (float, int):
    # return round(value * thresholds / max_value) * max_value / thresholds
    if type(value) == float or type(thresholds) == float or type(min_value) == float or type(max_value) == float:
        return_type = float
    else:
        return_type = int
    return return_type(round((value - min_value) * thresholds / (max_value - min_value)) * (max_value - min_value) / thresholds + min_value)


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
    print(Q_map(value=0.5, lowerLimit=0, upperLimit=1, scaledLowerLimit=0, scaledUpperLimit=100))
    assert(Q_map(value=0.5, lowerLimit=0, upperLimit=1, scaledLowerLimit=0, scaledUpperLimit=100) == 50)
    print(Q_map(value=0.9, lowerLimit=0, upperLimit=1, scaledLowerLimit=0, scaledUpperLimit=100))
    assert(Q_map(value=0.9, lowerLimit=0, upperLimit=1, scaledLowerLimit=0, scaledUpperLimit=100) == 90)

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

    print(50, Q_quantize(50, min_value=50, max_value=100, thresholds=5))
    assert(Q_quantize(50, min_value=50, max_value=100, thresholds=5) == 50)
    print(54, Q_quantize(54, min_value=50, max_value=100, thresholds=5))
    assert(Q_quantize(54, min_value=50, max_value=100, thresholds=5) == 50)
    print(58, Q_quantize(58, min_value=50, max_value=100, thresholds=5))
    assert(Q_quantize(58, min_value=50, max_value=100, thresholds=5) == 60)
    print(88, Q_quantize(88, min_value=50, max_value=100, thresholds=5))
    assert(Q_quantize(88, min_value=50, max_value=100, thresholds=5) == 90)
    print(98, Q_quantize(98, min_value=50, max_value=100, thresholds=5))
    assert(Q_quantize(98, min_value=50, max_value=100, thresholds=5) == 100)
    print(100, Q_quantize(100, min_value=50, max_value=100, thresholds=5))
    assert(Q_quantize(100, min_value=50, max_value=100, thresholds=5) == 100)


if __name__ == "__main__":
    main()
