import math
from Q_Functions import Q_Vector3D, Q_map
import numpy as np
import os


class Ray:
    def __init__(self, origin: Q_Vector3D, direction: Q_Vector3D):
        self.origin = origin
        self.direction = direction


class Primitive:
    """
    Use this as a test:
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    c = ((ray.origin - self.center).length ** 2) - (self.radius ** 2)
    """
    def __init__(self, center: Q_Vector3D):
        self.center = center


class CirclePrimitive(Primitive):
    def __init__(self, center: Q_Vector3D, radius: float):
        self.center = center
        self.radius = float(radius)

    def intersect(self, ray: Ray):
        b = 2 * ray.direction.dot_product(other_vector=(ray.origin - self.center))
        # c = ((ray.origin - self.center) * (ray.origin - self.center)).length - (self.radius ** 2)
        c = ((ray.origin - self.center).length ** 2) - (self.radius ** 2)

        # Test
        # ray_origin = np.array([ray.origin.x, ray.origin.y, ray.origin.z])
        # center = np.array([self.center.x, self.center.y, self.center.z])
        # radius = self.radius
        # c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2

        delta = b ** 2 - 4 * c
        if delta > 0:
            t1 = (-b + math.sqrt(delta)) / 2
            t2 = (-b - math.sqrt(delta)) / 2
            if t1 > 0 and t2 > 0:
                return min(t1, t2)
        return None


def sphere_intersect(center: Q_Vector3D, radius, ray: Ray):
    b = 2 * ray.direction.dot_product(other_vector=(ray.origin - center))
    c = ((ray.origin - center) * (ray.origin - center)).length - (radius ** 2)
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + math.sqrt(delta)) / 2
        t2 = (-b - math.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None


WIDTH = 64
HEIGHT = 48
SCREEN_RATIO = float(WIDTH) / float(HEIGHT)

camera_position = Q_Vector3D(0, 0, 0)

scene = [
    {'id': 1, 'item': CirclePrimitive(center=Q_Vector3D(x=0, y=0, z=5), radius=1.0)},
    {'id': 2, 'item': CirclePrimitive(center=Q_Vector3D(x=-2, y=-3, z=3), radius=1.0)}
]

# Camera is at 0, 0, 0
# Screen is at 0, 0, 1
# Objects are at 0, 0, 2

os.system('cls')

for y in range(HEIGHT):
    yy = Q_map(value=-y, lower_limit=-(HEIGHT - 1), upper_limit=0, scaled_lower_limit=-1.0, scaled_upper_limit=1.0)  # -((2 * y / float(HEIGHT - 1)) - 1)  # Q_map(value=-y, lower_limit=-(HEIGHT - 1), upper_limit=0, scaled_lower_limit=-1.0, scaled_upper_limit=1.0)  # (-y + (HEIGHT / 2.0)) / HEIGHT  # Need to make sure I did this right
    for x in range(WIDTH):
        xx = Q_map(value=x, lower_limit=0, upper_limit=WIDTH - 1, scaled_lower_limit=-1.0, scaled_upper_limit=1.0)  # (2 * x / float(WIDTH - 1)) - 1  # Q_map(value=x, lower_limit=0, upper_limit=WIDTH - 1, scaled_lower_limit=-1.0, scaled_upper_limit=1.0)  # (x - (WIDTH / 2.0)) / WIDTH
        # screen is on origin
        pixel = Q_Vector3D(xx, yy, 1)
        origin = camera_position
        direction = (pixel - origin).normalized()
        ray = Ray(origin=origin, direction=direction)

        if min(x, y) == 0 or x == WIDTH - 1 or y == HEIGHT - 1:
            print('#', end='')
            continue
        for object in scene:
            if object['item'].intersect(ray=ray):
                print('╝', end='')
                break
        else:
            print(' ', end='')

    print()


print()
ray_origin = np.array([0, 0, 0])
vOrigin = Q_Vector3D(x=0, y=0, z=0)
radius = 1.0
for x in range(5):
    for y in range(-3, 6):
        for z in range(2, 8):
            center = np.array([x, y, z])
            vCenter = Q_Vector3D(x=x, y=y, z=z)
            if np.linalg.norm(ray_origin - center) != (vOrigin - vCenter).length:
                print(np.linalg.norm(ray_origin - center), (vOrigin - vCenter).length, np.linalg.norm(ray_origin - center) == (vOrigin - vCenter).length)
