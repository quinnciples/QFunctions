import math


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
        else:
            print(type(self), type(other_object))
            raise('Unknown Type')

    #######################################
    #
    # Functions
    #
    #######################################

    def __str__(self):
        return f'Q_Vector3D(x={self.x}, y={self.y}, z={self.z}, length={self.length})'

    def __repr__(self):
        return f'Q_Vector3D(x={self.x}, y={self.y}, z={self.z})'


if __name__ == '__main__':

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
