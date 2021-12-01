import math
import random
from Q_Functions import Q_map


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


class Q_Vector3D:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def from_Vector3D(other_vector):
        return Q_Vector3D(x=other_vector.x, y=other_vector.y, z=other_vector.z)

    def dot_product(self, other_vector) -> float:
        return self.x * other_vector.x + self.y * other_vector.y + self.z * other_vector.z

    def element_wise_product(self, other_vector):
        return Q_Vector3D(x=self.x * other_vector.x, y=self.y * other_vector.y, z=self.z * other_vector.z)

    def cross_product(self, other_vector):
        return Q_Vector3D(x=self.y * other_vector.z - self.z * other_vector.y, y=self.z * other_vector.x - self.x * other_vector.z, z=self.x * other_vector.y - self.y * other_vector.x)

    @property
    def length(self):
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
        length = self.length
        if length > limit:
            scalar = limit / length
            self.x *= scalar
            self.y *= scalar
            self.z *= scalar

    def reflected(self, other_vector) -> float:
        # Maybe we should always normalize other_vector first?
        return Q_Vector3D.from_Vector3D(self - (other_vector * 2 * self.dot_product(other_vector=other_vector)))

    #######################################
    #
    # OPERATORS
    #
    #######################################

    def __add__(self, other_vector):
        return Q_Vector3D(x=self.x + other_vector.x, y=self.y + other_vector.y, z=self.z + other_vector.z)

    def __sub__(self, other_vector):
        return Q_Vector3D(x=self.x - other_vector.x, y=self.y - other_vector.y, z=self.z - other_vector.z)

    def __mul__(self, other_object):
        if type(other_object) in (float, int):
            return Q_Vector3D(x=self.x * other_object, y=self.y * other_object, z=self.z * other_object)
        elif type(other_object) == Q_Vector3D:
            return self.element_wise_product(other_vector=other_object)
        return None

    #######################################
    #
    # Functions
    #
    #######################################

    def __str__(self):
        return f'Q_Vector3D(x={self.x}, y={self.y}, z={self.z}, length={self.length})'

    def __repr__(self):
        return f'Q_Vector3D(x={self.x}, y={self.y}, z={self.z})'


test_vector = Q_Vector3D(x=1, y=2, z=3)
print(test_vector.length)
print(math.sqrt(test_vector.x ** 2 + test_vector.y ** 2 + test_vector.z ** 2))
print(1 ** 2 + 2 ** 2 + 3 ** 2, math.sqrt(1 ** 2 + 2 ** 2 + 3 ** 2))
print('Mult test')
print('Scalar mult')
test_vector = Q_Vector3D(x=1, y=1, z=1)
print(test_vector)
new_vector = test_vector * 2.0
print(new_vector)
print('Vector mult')
test_vector = Q_Vector3D(x=5, y=4, z=3)
mult_vector = Q_Vector3D(x=2, y=3, z=4)
print(test_vector, mult_vector)
new_vector = test_vector * mult_vector
print(new_vector)
print('Limit test')
test_vector = Q_Vector3D(x=5, y=4, z=3)
print(test_vector)
test_vector.limit(10)
print(f'Limited to 10: {test_vector}')
test_vector.limit(5)
print(f'Limited to 5: {test_vector}')
print('Reflected test')
test_vector = Q_Vector3D(x=0, y=0, z=2)
print('Initial vector ', test_vector)
surface_vector = Q_Vector3D(x=0, y=0, z=-1).normalized()
print('Surface vector ', surface_vector)
reflected_vector = test_vector.reflected(surface_vector)
print('Reflected vector ', reflected_vector)

# https://github.com/OmarAflak/RayTracer-CPP/blob/master/main.cpp
