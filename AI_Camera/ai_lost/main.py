from collections import defaultdict, deque
import numpy as np
import time
from copy import deepcopy

from AI_Camera.core.main import BaseModule


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Checking if a point is inside a polygon
def point_in_polygon(point, polygon):
    """https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
    """
    num_vertices = len(polygon)
    x, y = point.x, point.y
    inside = False

    p1 = polygon[0]

    for i in range(1, num_vertices + 1):
        p2 = polygon[i % num_vertices]

        if y > min(p1.y, p2.y):
            if y <= max(p1.y, p2.y):
                if x <= max(p1.x, p2.x):
                    x_intersection = (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x

                    if p1.x == p2.x or x <= x_intersection:
                        inside = not inside
        p1 = p2

    return inside


class AILost(BaseModule):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.tracklets_history = defaultdict(lambda: deque(maxlen=self.cfg.checker.track_history_len))
        self.is_not_moving_history = defaultdict(
                                        lambda: deque(maxlen=self.cfg.checker.is_not_moving_checker_len)
                                    )
        self.last_appear = defaultdict(int)
        self.is_lost = {}
        self.is_any_person = False
    

    def setup_management_region(self, points):
        self.manage_region = []
        for point in points:
            self.manage_region.append([Point(point[0], point[1])])    
         

    def check_inside_region(self, point):
        return point_in_polygon(Point(point[0], point[1]), self.manage_region)
    

    def track_proposal_lost_object(self, image):
        dets = self.detector.do_detect(image)[0]
        tracklets = self.tracker.do_track(dets=dets)
        bottom_center_points = np.concatenate(
                                [(tracklets[:, 0:1] + tracklets[:, 2:3]) / 2,
                                 tracklets[:, 3:4]], axis=1)

        self.is_any_person = False
        for i, track_ in enumerate(tracklets):
            if track_[-2] == 0:
                is_inside_management_region = self.check_inside_region(bottom_center_points[i])
                if is_inside_management_region:
                    self.is_any_person = True
            elif track_[-2] != 0: # do not track human
                is_inside_management_region = self.check_inside_region(bottom_center_points[i])
                tid = track_[-1]
                self.tracklets_history[tid].append((track_[:4]), bottom_center_points[i])
                self.last_appear[tid] = time.time()


    def find_items_abandon(self):
        abandon_objects_dict = {}
        if not self.is_any_person:
            for (tid, track_hist) in self.tracklets_history.items():
                if len(track_hist) < self.cfg.checker.track_history_len:
                    continue

                coor_start = track_hist[1][0]
                coor_end = track_hist[1][-1]

                if np.linalg.norm(coor_end - coor_start) < self.cfg.checker.is_moving_threshold:
                    self.is_not_moving_history[tid].append(1)
                else:
                    self.is_not_moving_history[tid].append(0)

                if len(self.is_not_moving_history[tid]) == self.cfg.checker.is_not_moving_checker_len:
                    point_ = sum(self.is_not_moving_history[tid])

                    if point_ > self.cfg.checker.is_not_moving_checker_len * 0.9 + 1:
                        abandon_objects_dict[tid] = track_hist[0]
        
        return abandon_objects_dict
        
    
    def remove_disappeared_objects(self):
        current_time = time.time()
        for tid in self.last_appear:
            last_time = self.last_appear[tid]
            if current_time - last_time > 3:
                del self.last_appear[tid]
                del self.tracklets_history[tid]
                del self.is_not_moving_history[tid]
    

    def run(self, image):
        self.track_proposal_lost_object(image)
        self.remove_disappeared_objects()
        abandon_dict = self.find_items_abandon()

        return abandon_dict



            



            