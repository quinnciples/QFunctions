import math
from Q_Functions import Q_Vector3D
from Q_Functions import Q_map


def sphere_intersect(center: Q_Vector3D, radius, ray_origin: Q_Vector3D, ray_direction: Q_Vector3D):
    b = 2 * ray_direction.dot_product(other_vector=(ray_origin - center))
    c = ((ray_origin - center) * (ray_origin - center)).length - (radius ** 2)
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
circle_center = Q_Vector3D(0, 0, 3)
circle_radius = 1.5

# Camera is at 0, 0, 0
# Screen is at 0, 0, 1
# Objects are at 0, 0, 2

for y in range(HEIGHT):
    yy = Q_map(value=y, lower_limit=0, upper_limit=HEIGHT, scaled_lower_limit=-1, scaled_upper_limit=1)  # (-y + (HEIGHT / 2.0)) / HEIGHT  # Need to make sure I did this right
    for x in range(WIDTH):
        xx = Q_map(value=x, lower_limit=0, upper_limit=WIDTH, scaled_lower_limit=-1, scaled_upper_limit=1)  # (x - (WIDTH / 2.0)) / WIDTH
        # screen is on origin
        pixel = Q_Vector3D(xx, yy, 1)
        origin = camera_position
        direction = (pixel - origin).normalized()

        if min(x, y) == 0 or x == WIDTH - 1 or y == HEIGHT - 1:
            print('#', end='')
        elif sphere_intersect(center=circle_center, radius=circle_radius, ray_origin=origin, ray_direction=direction):
            print('╝', end='')
        else:
            print(' ', end='')
    print()
