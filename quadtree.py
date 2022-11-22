from collections import namedtuple
from typing import List, Optional, Self, Iterable
from random import random, seed
from math import sqrt
from time import perf_counter

Point = namedtuple('Point', ['x', 'y'])


class QuadTree:
    def __init__(self, center: Point, width: float, capacity: int):
        self.center: Point = center
        self.width: float = float(width)
        self.capacity: int = capacity
        self.items: List[Point] = list()
        self.is_split: bool = False
        self.upper_left_qt: Optional[Self | None] = None
        self.upper_right_qt: Optional[Self | None] = None
        self.lower_left_qt: Optional[Self | None] = None
        self.lower_right_qt: Optional[Self | None] = None

    def __repr__(self):
        return f'QuadTree(center={self.center}, width={self.width}, capacity={self.capacity})'

    def split(self):  # sourcery skip: raise-specific-error
        # print("Splitting...")
        if self.is_split:
            raise Exception('Can not split an already split QuadTree.')
        self.is_split = True
        split_width: float = self.width / 2.0
        self.upper_left_qt = QuadTree(
            Point(x=self.center.x - split_width, y=self.center.y + split_width),
            width=split_width,
            capacity=self.capacity,
        )
        self.upper_right_qt = QuadTree(
            Point(x=self.center.x + split_width, y=self.center.y + split_width),
            width=split_width,
            capacity=self.capacity,
        )
        self.lower_left_qt = QuadTree(
            Point(x=self.center.x - split_width, y=self.center.y - split_width),
            width=split_width,
            capacity=self.capacity,
        )
        self.lower_right_qt = QuadTree(
            Point(x=self.center.x + split_width, y=self.center.y - split_width),
            width=split_width,
            capacity=self.capacity,
        )
        while self.items:
            point: Point = self.items.pop()
            results: list = [
                self.upper_left_qt.assign(point=point),
                self.upper_right_qt.assign(point=point),
                self.lower_left_qt.assign(point=point),
                self.lower_right_qt.assign(point=point),
            ]
            split_assign: bool = any(results)
            if not split_assign:
                raise Exception(f'Unable to assign {point} during the split process.')

    def assign(self, point: Point) -> bool:
        if self.is_split:
            return any(
                (
                    self.upper_left_qt.assign(point=point),
                    self.upper_right_qt.assign(point=point),
                    self.lower_left_qt.assign(point=point),
                    self.lower_right_qt.assign(point=point),
                )
            )
        # print(f'This QT has a center of {self.center} and width of {self.width} -- trying to assign point {point}')
        # print(f'{self.center=}   {self.width=}')
        # print(f'{(self.center.x - self.width)=}   {(self.center.x + self.width)=}')
        # print(f'{(self.center.y - self.width)=}   {(self.center.y + self.width)=}')
        # print(f'{(self.center.x - self.width) <= point.x <= (self.center.x + self.width)=}')
        # print(f'{(self.center.y - self.width) <= point.y <= (self.center.y + self.width)=}')

        if ((self.center.x - self.width) <= point.x < (self.center.x + self.width)) and (
            (self.center.y - self.width) <= point.y < (self.center.y + self.width)
        ):
            self.items.append(point)
            if len(self.items) >= self.capacity:
                self.split()
            return True

        return False

    def contains(self, point: Point) -> bool:
        if self.is_split:
            return any(
                (
                    self.upper_left_qt.contains(point=point),
                    self.upper_right_qt.contains(point=point),
                    self.lower_left_qt.contains(point=point),
                    self.lower_right_qt.contains(point=point),
                )
            )
        return point in self.items

    def get_all_in_quadrant(self, point: Point) -> Iterable[Point]:
        if not self.is_split:
            yield from self.items
            return
        if point.x <= self.center.x:
            if point.y >= self.center.y:
                yield from self.upper_left_qt.get_all_in_quadrant(point=point)
            if point.y <= self.center.y:
                yield from self.lower_left_qt.get_all_in_quadrant(point=point)
        if point.x >= self.center.x:
            if point.y >= self.center.y:
                yield from self.upper_right_qt.get_all_in_quadrant(point=point)
            if point.y <= self.center.y:
                yield from self.lower_right_qt.get_all_in_quadrant(point=point)
        return

    def get_nearby(self, point: Point, distance: float) -> Iterable[Point]:
        if not self.is_split:
            dist_squared: float = distance**2
            yield from [
                item for item in self.items if ((item.x - point.x) ** 2 + (item.y - point.y) ** 2) <= dist_squared
            ]
            return
        if distance < 0:
            return
        if point.x - distance <= self.center.x:
            if point.y + distance >= self.center.y:
                yield from self.upper_left_qt.get_nearby(point=point, distance=distance)
            if point.y - distance <= self.center.y:
                yield from self.lower_left_qt.get_nearby(point=point, distance=distance)
        if point.x + distance >= self.center.x:
            if point.y + distance >= self.center.y:
                yield from self.upper_right_qt.get_nearby(point=point, distance=distance)
            if point.y - distance <= self.center.y:
                yield from self.lower_right_qt.get_nearby(point=point, distance=distance)
        return


if __name__ == '__main__':
    p1 = Point(1.0, 1.0)
    p2 = Point(-1.0, -1.0)
    p3 = Point(0.25, 0.01)
    qt = QuadTree(Point(0, 0), 5, capacity=5)
    qt.assign(p1)
    qt.assign(p2)
    qt.assign(p3)
    print('*** items in this tree')
    print(qt.items)
    print('*' * 20)
    print('*** get_all_in_quadrant test before split')
    for pt in qt.get_all_in_quadrant(p1):
        print(pt)
    print('*' * 20)
    print('*** get_nearby test before split with distance=10.0')
    for pt in qt.get_nearby(Point(0, 0), distance=10.0):
        print(pt)
    print('*' * 20)
    print('*** get_nearby test before split with distance=0.6')
    for pt in qt.get_nearby(Point(0, 0), distance=0.6):
        print(pt)
    print('*' * 20)
    qt.split()
    print('*** breakdown of items in split trees')
    print(f'upper left {qt.upper_left_qt.items}')
    print(f'upper right {qt.upper_right_qt.items}')
    print(f'lower left {qt.lower_left_qt.items}')
    print(f'lower right{qt.lower_right_qt.items}')
    print('*' * 20)
    print('*** get_all_in_quadrant test after split')
    for pt in qt.get_all_in_quadrant(p1):
        print(pt)
    print('*** get_nearby test after split with poiont (0, 0) and distance=10.0')
    for pt in qt.get_nearby(Point(0, 0), distance=10.0):
        print(pt)
    print('*' * 20)
    print('*** get_nearby test after split with point (0, 0) and distance=0.6')
    for pt in qt.get_nearby(Point(0, 0), distance=0.6):
        print(pt)
    print('*' * 20)
    print('*** Assign 1,000 points test')
    del qt
    qt2 = QuadTree(center=Point(0, 0), width=2, capacity=25)
    seed(123)
    for i in range(1, 1_001):
        print(f'\r{i}', end='', flush=True)
        qt2.assign(Point(random() * 2.0 - 1.0, random() * 2.0 - 1.0))
    print()
    print('*' * 20)
    print('Distance calculation tests - the biggie')
    number_of_points: int = 10_000
    quadtree_capacity: int = 100
    point_list: list = [Point(random() * 20.0 - 10.0, random() * 20.0 - 10.0) for _ in range(number_of_points)]

    def is_nearby(this_point: Point, other_point: Point, distance_threshold: float) -> bool:
        return sqrt((this_point.x - other_point.x) ** 2 + (this_point.y - other_point.y) ** 2) <= distance_threshold

    results: dict = {}
    starting_time = perf_counter()
    print('Starting brute force method...')
    for idx, point in enumerate(point_list[:]):
        print(f'\r{idx}/{number_of_points}', end='', flush=True)
        results[point] = set()
        for other_point in point_list[:]:
            if point is not other_point and is_nearby(
                this_point=point, other_point=other_point, distance_threshold=0.25
            ):
                results[point].add(other_point)
    print()
    print(f'Process took {perf_counter() - starting_time} to complete..')
    starting_time = perf_counter()
    quadtree_test = QuadTree(center=Point(0, 0), width=10.1, capacity=quadtree_capacity)
    for idx, point in enumerate(point_list):
        print(f'\rAdding {idx}/{number_of_points} to QuadTree...', end='', flush=True)
        quadtree_test.assign(point=point)
    print()
    print(f'Process took {perf_counter() - starting_time} to complete..')
    results_qt: dict = {}
    starting_time = perf_counter()
    print('Starting QuadTree method...')
    for idx, point in enumerate(point_list):
        print(f'\r{idx}/{number_of_points}', end='', flush=True)
        results_qt[point] = set()
        for other_point in quadtree_test.get_nearby(point=point, distance=0.3):
            if point is not other_point and is_nearby(
                this_point=point, other_point=other_point, distance_threshold=0.25
            ):
                results_qt[point].add(other_point)
    print()
    print(f'Process took {perf_counter() - starting_time} to complete..')
    print(results == results_qt)
    for point in point_list:
        if len(results[point]) != len(results_qt[point]):
            print(f'{point} *** {results[point]}              *** {results_qt[point]}')
