from collections import defaultdict, deque
import numpy as np
import time
import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from omegaconf import OmegaConf

from AI_Camera.core.main import BaseModule


# class Point:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

# # Checking if a point is inside a polygon
# def point_in_polygon(point, polygon):
#     """https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
#     """
#     num_vertices = len(polygon)
#     x, y = point.x, point.y
#     inside = False

#     p1 = polygon[0]

#     for i in range(1, num_vertices + 1):
#         p2 = polygon[i % num_vertices]

#         if y > min(p1.y, p2.y):
#             if y <= max(p1.y, p2.y):
#                 if x <= max(p1.x, p2.x):
#                     x_intersection = (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x
#                     if p1.x == p2.x or x <= x_intersection:
#                         inside = not inside
#         p1 = p2

#     return inside

class ViewTransformer:
    def __init__(self, source: np.ndarray) -> None:
        self.m = self.cal_transform_matrix(source)

    def cal_transform_matrix(self, source):
        x1 = np.min(source[:, 0])
        x2 = np.max(source[:, 0])
        y1 = np.min(source[:, 1])
        y2 = np.max(source[:, 1])

        new_source = np.array([(x1, y1), (x1, y2), (x2, y2), (x2, y1)], dtype=np.float32)
        target = np.array([(0, 0), (x2-x1, 0), (x2-x1, y2-y1), (0, y2-y1)], dtype=np.float32)
        m = cv2.getPerspectiveTransform(new_source, target)

        return m

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


class AILost(BaseModule):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.tracklets_history = defaultdict(lambda: deque(maxlen=self.cfg.checker.track_history_len))
        self.is_not_moving_history = defaultdict(
                                        lambda: deque(maxlen=self.cfg.checker.is_not_moving_checker_len)
                                    )
        self.last_appear = defaultdict(int)
        self.is_lost = defaultdict(lambda: False)
        self.is_any_person = False

        points = OmegaConf.to_object(self.cfg.points)
        self.setup_management_region(points)
    

    def setup_management_region(self, points):
        points = np.array(points).astype(np.float32)
        self.vt = ViewTransformer(points)
        bird_view_manage_region = self.vt.transform_points(points)
        self.manage_region = []

        for point in bird_view_manage_region:
            self.manage_region.append((point[0], point[1]))
        self.manage_region = Polygon(self.manage_region)


    def check_inside_region(self, point):
        point = Point(point[0], point[1])
        return self.manage_region.contains(point)
    

    def track_proposal_lost_object(self, image):
        dets = self.detector.do_detect(image)[0]
        tracklets = self.tracker.do_track(dets=dets)

        self.tracklets = tracklets
        
        self.is_any_person = False

        if tracklets.shape[0] > 0:
            bottom_center_points = np.concatenate(
                                    [(tracklets[:, 0:1] + tracklets[:, 2:3]) / 2,
                                      tracklets[:, 3:4]], axis=1)
            bottom_center_points = self.vt.transform_points(bottom_center_points)

            for i, track_ in enumerate(tracklets):
                if track_[-2] == 0: # if class is person
                    is_inside_management_region = self.check_inside_region(bottom_center_points[i])
                    if is_inside_management_region:
                        self.is_any_person = True
                elif track_[-2] != 0: 
                    is_inside_management_region = self.check_inside_region(bottom_center_points[i])
                    if is_inside_management_region:
                        tid = track_[-1]
                        self.tracklets_history[tid].append((track_[:4], bottom_center_points[i]))
                        self.last_appear[tid] = time.time()
                        

    def find_lost_objects(self):
        abandon_objects_dict = {}
        for track in self.tracklets:
            tid = track[-1]
            track_hist = self.tracklets_history[tid]
            if len(track_hist) < self.cfg.checker.track_history_len:
                continue

            coor_start = track_hist[0][1]
            coor_end = track_hist[-1][1]

            if np.linalg.norm(coor_end - coor_start) < self.cfg.checker.is_moving_threshold:
                self.is_not_moving_history[tid].append(1)
            else:
                self.is_not_moving_history[tid].append(0)

            if len(self.is_not_moving_history[tid]) == self.cfg.checker.is_not_moving_checker_len:
                point_ = sum(self.is_not_moving_history[tid])

                if point_ > self.cfg.checker.is_not_moving_checker_len * 0.8:
                    if not self.is_any_person:
                        if not self.is_lost[tid]:
                            self.is_lost[tid] = True
                        abandon_objects_dict[tid] = track_hist[-1][0]
                    elif self.is_lost[tid]:
                        abandon_objects_dict[tid] = track_hist[-1][0]
                else:
                    self.is_lost[tid] = False

        return abandon_objects_dict
        
    
    def remove_disappeared_objects(self):
        current_time = time.time()
        for tid in list(self.last_appear.keys()):
            last_time = self.last_appear[tid]

            if (current_time - last_time) > 10: # in second
                del self.last_appear[tid]
                del self.tracklets_history[tid]

                if tid in self.is_not_moving_history:
                    del self.is_not_moving_history[tid]
                
                if tid in self.is_lost:
                    del self.is_lost[tid]
    

    def run(self, image):
        self.track_proposal_lost_object(image)
        abandon_dict = self.find_lost_objects()
        self.remove_disappeared_objects()

        return abandon_dict


    def __call__(self, image):
        return self.run(image)



            



            