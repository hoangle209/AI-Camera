import cv2
import numpy as np
from collections import defaultdict, deque

from AI_Camera.core.main import BaseModule
from AI_Camera.utils import get_pylogger
logger = get_pylogger()

SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250
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
    
    def __init__(self, config) -> None:
        super().__init__(config)

        self.view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
        self.coordinates = defaultdict(lambda: deque(maxlen=self.fps))

    def reset(self):
        self.coordinates = defaultdict(lambda: deque(maxlen=self.fps))

    def estimate_speed(self, frame):
        labels = []
        dets = self.detector.do_detect(frame)[0]
        tracklets = self.tracker.do_track(dets=dets)

        bottom_center_points = np.concatenate(
                                [(tracklets[:, 0:1] + tracklets[:, 2:3]) / 2,
                                 tracklets[:, 3:4]], axis=1)
        points = self.view_transformer.transform_points(points=bottom_center_points).astype(int)
        track_id = tracklets[:, -1]

        for t_id, (_, y) in zip(track_id, points):
            self.coordinates[t_id].append(y)
            if len(self.coordinates[t_id]) < self.fps / 2:
                labels.append(f"#{t_id}")
            else:
                coordinate_start = self.coordinates[t_id][-1]
                coordinate_end = self.coordinates[t_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(self.coordinates[t_id]) / self.fps
                speed = distance / time * 3.6
                labels.append(f"#{t_id} {int(speed)} km/h")

        return tracklets, labels
    

    def __call__(self, frame):
        return self.estimate_speed(frame)

    