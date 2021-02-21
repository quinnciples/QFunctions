from noise import pnoise2


class Q_MinMaxScaler:
    def __init__(self, items):
        self.input = items
        self.minimum = min(items)
        self.maximum = max(items)
        self.scale = self.maximum - self.minimum

    def transform(self):
        output = [Q_map(x, self.minimum, self.maximum, 0, 1) for x in self.input]
        return output

    def reverse_transform(self, values):
        output = [Q_map(x, 0, 1, self.minimum, self.maximum) for x in values]
        return output


def Q_constrain(val, lowerLimit=0, upperLimit=1):
    if val < lowerLimit:
        return lowerLimit
    elif val > upperLimit:
        return upperLimit
    else:
        return val


def Q_map(val, lowerLimit, upperLimit, newLowerLimit, newUpperLimit):
    temp_val = val - lowerLimit
    temp_scale = temp_val / (upperLimit - lowerLimit)
    temp_output = ((newUpperLimit - newLowerLimit) * temp_scale) + newLowerLimit
    return temp_output


def Q_quantize(value: (float, int), max_value: (float, int), thresholds: (float, int), min_value: (float, int) = 0) -> (float, int):
    # return round(value * thresholds / max_value) * max_value / thresholds
    if type(value) == float or type(thresholds) == float or type(min_value) == float or type(max_value) == float:
        return_type = float
    else:
        return_type = int
    return return_type(round((value - min_value) * thresholds / (max_value - min_value)) * (max_value - min_value) / thresholds + min_value)


def main():
    items = [0, 1, 2, 3, 4, 5]
    print(items)
    Q_scale = Q_MinMaxScaler(items)
    transformed = Q_scale.transform()
    print(transformed)
    reverted = Q_scale.reverse_transform(transformed)
    print(reverted)
    fake_data = [pnoise2(x / 9, 1.678) for x in range(5)]
    print(fake_data)
    fake_reverted = Q_scale.reverse_transform(fake_data)
    print(fake_reverted)
    print(0, Q_quantize(0, 100, 4))
    print(1, Q_quantize(1, 100, 4))
    print(20, Q_quantize(20, 100, 4))
    print(26, Q_quantize(26, 100, 4))
    print(45, Q_quantize(45, 100, 4))
    print(66, Q_quantize(66, 100, 4))
    print(75, Q_quantize(75, 100, 4))
    print(99, Q_quantize(99, 100, 4))
    print(100, Q_quantize(100, 100, 4))
    print(50, Q_quantize(50, min_value=50, max_value=100, thresholds=5))
    print(54, Q_quantize(54, min_value=50, max_value=100, thresholds=5))
    print(58, Q_quantize(58, min_value=50, max_value=100, thresholds=5))
    print(88, Q_quantize(88, min_value=50, max_value=100, thresholds=5))
    print(98, Q_quantize(98, min_value=50, max_value=100, thresholds=5))
    print(100, Q_quantize(100, min_value=50, max_value=100, thresholds=5))


if __name__ == "__main__":
    main()
