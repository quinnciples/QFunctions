import math
from Q_Functions import Q_Vector3d, Q_map
import numpy as np
import os
import matplotlib.pyplot as plt


class Ray:
    def __init__(self, origin: Q_Vector3d, direction: Q_Vector3d):
        self.origin = origin
        self.direction = direction


class Scene:
    def __init__(self, objects: list = [], lights: list = []):
        self.objects = objects
        self.lights = lights

    def nearest_intersection(self, ray: Ray):
        min_distance = math.inf
        obj = None
        for _, object in enumerate(self.objects):
            intersection = object['item'].intersect(ray=ray)
            if intersection and intersection < min_distance:
                min_distance = intersection
                obj = object

        return obj, min_distance

    def render(self, camera_position: Q_Vector3d, width: int = 64, height: int = 64):
        image = np.zeros((height, width, 3))
        SCREEN_RATIO = float(width) / float(height)
        SCREEN_DIMS = {'left': -1, 'top': 1 / SCREEN_RATIO, 'right': 1, 'bottom': -1 / SCREEN_RATIO}

        for y in range(height):
            print(f'\r{y + 1}/{height}', end='')
            yy = Q_map(value=-y, lower_limit=-(height - 1), upper_limit=0, scaled_lower_limit=SCREEN_DIMS['bottom'], scaled_upper_limit=SCREEN_DIMS['top'])  # -((2 * y / float(HEIGHT - 1)) - 1)  # Q_map(value=-y, lower_limit=-(HEIGHT - 1), upper_limit=0, scaled_lower_limit=-1.0, scaled_upper_limit=1.0)  # (-y + (HEIGHT / 2.0)) / HEIGHT  # Need to make sure I did this right
            for x in range(width):
                xx = Q_map(value=x, lower_limit=0, upper_limit=width - 1, scaled_lower_limit=-1.0, scaled_upper_limit=1.0)  # (2 * x / float(WIDTH - 1)) - 1  # Q_map(value=x, lower_limit=0, upper_limit=WIDTH - 1, scaled_lower_limit=-1.0, scaled_upper_limit=1.0)  # (x - (WIDTH / 2.0)) / WIDTH
                pixel = Q_Vector3d(xx, yy, 0)

                origin = camera_position
                direction = (pixel - origin).normalized()
                ray = Ray(origin=origin, direction=direction)

                nearest_object, distance_to_object = scene.nearest_intersection(ray=ray)

                if nearest_object is None:
                    continue

                # Lighting
                object_color = nearest_object['item'].color

                intersection_point = origin + direction * distance_to_object
                normal_to_surface = (intersection_point - nearest_object['item'].position).normalized()
                shifted_point = intersection_point + normal_to_surface * 1e-5
                direction_from_intersection_to_light = (self.lights[0]['position'] - shifted_point).normalized()

                ray = Ray(origin=shifted_point, direction=direction_from_intersection_to_light)
                nearest_object, distance_to_object = scene.nearest_intersection(ray=ray)

                distance_to_light = (self.lights[0]['position'] - intersection_point).length
                is_shadowed = distance_to_object < distance_to_light

                intensity = math.fabs(direction_from_intersection_to_light.dot_product(normal_to_surface))
                object_color = (object_color[0] * self.lights[0]['color'][0] * intensity, object_color[1] * self.lights[0]['color'][1] * intensity, object_color[2] * self.lights[0]['color'][2] * intensity)

                if is_shadowed:
                    continue
                else:
                    image[y, x] = object_color  # nearest_object['color'] if nearest_object else (0, 0, 0)

        plt.imsave('image.png', image)
        print()


class Primitive:
    def __init__(self, center: Q_Vector3d, color: tuple):
        self.position = center
        self.color = color


class SpherePrimitive(Primitive):
    """
    Use this as a test:
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    c = ((ray.origin - self.center).length ** 2) - (self.radius ** 2)
    """
    def __init__(self, center: Q_Vector3d, color: tuple, radius: float):
        # self.center = center
        Primitive.__init__(self, center=center, color=color)
        self.radius = float(radius)

    def intersect(self, ray: Ray):
        b = 2 * ray.direction.dot_product(other_vector=(ray.origin - self.position))
        c = ((ray.origin - self.position).length ** 2) - (self.radius ** 2)

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
CAMERA = Q_Vector3d(0, 0, -1.75)

objects = [
    {'item': SpherePrimitive(center=Q_Vector3d(x=2.5, y=0, z=10), color=(1, 0, 1), radius=3)},
    {'item': SpherePrimitive(center=Q_Vector3d(x=-4.5, y=0, z=10), color=(0, 1, 1), radius=2.0)},
    # {'id': 3, 'item': SpherePrimitive(center=Q_Vector3d(x=-0, y=-1000, z=0), radius=990.0)},
]

lights = [
    {'position': Q_Vector3d(0, 10, 5), 'color': (1, 1, 1)}
]

scene = Scene(objects=objects, lights=lights)

os.system('cls')
print()

scene.render(camera_position=CAMERA, width=WIDTH, height=HEIGHT)

# Test
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
