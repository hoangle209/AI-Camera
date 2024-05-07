import cv2
import numpy as np
from collections import defaultdict, deque
import time

from AI_Camera.core.main import BaseModule
from AI_Camera.utils import get_pylogger
logger = get_pylogger()

SOURCE = np.array([[283, 83], [529, 58], [369, 277], [120, 275]])

TARGET_WIDTH = 3.4
TARGET_HEIGHT = 3.15
TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


class SpeedEstimation(BaseModule):
    fps = 30
    
    def __init__(self, config, source=SOURCE, target=TARGET) -> None:
        super().__init__(config)

        self.view_transformer = ViewTransformer(source=source, target=target)
        self.coordinates = defaultdict(lambda: deque(maxlen=self.fps))
        self.last_appear = {}

    def reset(self):
        self.coordinates = defaultdict(lambda: deque(maxlen=self.fps))

    def estimate_speed(self, frame):
        speed = {}
        labels = []
        dets = self.detector.do_detect(frame)[0]
        tracklets = self.tracker.do_track(dets=dets)

        if tracklets.shape[0] == 0:
            return np.array([]), [], {}

        bottom_center_points = np.concatenate(
                                [(tracklets[:, 0:1] + tracklets[:, 2:3]) / 2,
                                 tracklets[:, 3:4]], axis=1)
        points = self.view_transformer.transform_points(points=bottom_center_points)
        track_id = tracklets[:, -1]

        for t_id, (_, y), tracklet in zip(track_id, points, tracklets):
            self.last_appear[t_id] = time.time()
            self.coordinates[t_id].append(y)

            tracklet[:4][tracklet[:4]<0] = 0
            if len(self.coordinates[t_id]) < self.fps / 2:
                labels.append(f"#{t_id}")
                speed[f"{t_id}"] = (tracklet[:4], None)
            else:
                coordinate_start = self.coordinates[t_id][-1]
                coordinate_end = self.coordinates[t_id][0]
                distance = abs(coordinate_start - coordinate_end)
                obj_time = len(self.coordinates[t_id]) / self.fps
                obj_speed = distance / obj_time * 3.6
                labels.append(f"#{t_id} {int(obj_speed)} km/h")
                speed[f"{t_id}"] = (tracklet[:4], round(obj_speed, 1))

        self.remove_disappeared_objects()

        return tracklets, labels, speed
    

    def remove_disappeared_objects(self):
        current_time = time.time()
        for tid in list(self.last_appear.keys()):
            last_time = self.last_appear[tid]

            if (current_time - last_time) > 5: # in second
                del self.last_appear[tid]

                if tid in self.coordinates[tid]:
                    del self.coordinates[tid]
    

    def __call__(self, frame):
        return self.estimate_speed(frame)

    