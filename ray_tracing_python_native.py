import math
from Q_Functions import Q_Vector3D, Q_map
import numpy as np
import os
import matplotlib.pyplot as plt


class Ray:
    def __init__(self, origin: Q_Vector3D, direction: Q_Vector3D):
        self.origin = origin
        self.direction = direction


class Primitive:
    def __init__(self, center: Q_Vector3D):
        self.center = center


class CirclePrimitive(Primitive):
    """
    Use this as a test:
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    c = ((ray.origin - self.center).length ** 2) - (self.radius ** 2)
    """
    def __init__(self, center: Q_Vector3D, radius: float):
        self.center = center
        self.radius = float(radius)

    def intersect(self, ray: Ray):
        b = 2 * ray.direction.dot_product(other_vector=(ray.origin - self.center))
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


WIDTH = 640
HEIGHT = 480
SCREEN_RATIO = float(WIDTH) / float(HEIGHT)
SCREEN_DIMS = {'left': -1, 'top': 1 / SCREEN_RATIO, 'right': 1, 'bottom': -1 / SCREEN_RATIO}
camera_position = Q_Vector3D(0, 0, -1.75)

scene = [
    {'id': 1, 'item': CirclePrimitive(center=Q_Vector3D(x=2.5, y=0, z=10), radius=1.5)},
    {'id': 2, 'item': CirclePrimitive(center=Q_Vector3D(x=-4.5, y=-2.5, z=25.0), radius=1.0)},
    {'id': 3, 'item': CirclePrimitive(center=Q_Vector3D(x=-0, y=-1000, z=0), radius=990.0)},
]

os.system('cls')
print()
image = np.zeros((HEIGHT, WIDTH, 3))

for y in range(HEIGHT):
    print(f'\r{y + 1}/{HEIGHT}', end='')
    yy = Q_map(value=-y, lower_limit=-(HEIGHT - 1), upper_limit=0, scaled_lower_limit=SCREEN_DIMS['bottom'], scaled_upper_limit=SCREEN_DIMS['top'])  # -((2 * y / float(HEIGHT - 1)) - 1)  # Q_map(value=-y, lower_limit=-(HEIGHT - 1), upper_limit=0, scaled_lower_limit=-1.0, scaled_upper_limit=1.0)  # (-y + (HEIGHT / 2.0)) / HEIGHT  # Need to make sure I did this right
    for x in range(WIDTH):
        xx = Q_map(value=x, lower_limit=0, upper_limit=WIDTH - 1, scaled_lower_limit=-1.0, scaled_upper_limit=1.0)  # (2 * x / float(WIDTH - 1)) - 1  # Q_map(value=x, lower_limit=0, upper_limit=WIDTH - 1, scaled_lower_limit=-1.0, scaled_upper_limit=1.0)  # (x - (WIDTH / 2.0)) / WIDTH
        # screen is on origin
        pixel = Q_Vector3D(xx, yy, 0)
        origin = camera_position
        direction = (pixel - origin).normalized()
        ray = Ray(origin=origin, direction=direction)

        for object in scene:
            intersection = object['item'].intersect(ray=ray)
            if intersection:
                image[y, x] = (1, 1, 1)
                break
        else:
            image[y, x] = (0, 0, 0)

plt.imsave('image.png', image)
print()

# ray_origin = np.array([0, 0, 0])
# vOrigin = Q_Vector3D(x=0, y=0, z=0)
# radius = 1.0
# for x in range(5):
#     for y in range(-3, 6):
#         for z in range(2, 8):
#             center = np.array([x, y, z])
#             vCenter = Q_Vector3D(x=x, y=y, z=z)
#             if np.linalg.norm(ray_origin - center) != (vOrigin - vCenter).length:
#                 print(np.linalg.norm(ray_origin - center), (vOrigin - vCenter).length, np.linalg.norm(ray_origin - center) == (vOrigin - vCenter).length)
