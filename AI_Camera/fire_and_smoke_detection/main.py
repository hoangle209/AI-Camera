from collections import defaultdict, deque
import time

from AI_Camera.core.main import BaseModule
from AI_Camera.utils import get_pylogger
logger = get_pylogger()


class FireSmokeDetection(BaseModule):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.tracked_fire_and_smoke = defaultdict(lambda: deque(maxlen=self.cfg.inspector.confirm_thresh))
        self.last_appear = {}

    def detect_fire_and_smoke(self, batch):
        dets = self.detector.do_detect(batch)
        return dets
    
    def track_fire_and_smoke(self, dets):
        tracklets = self.tracker.do_track(dets=dets)
        for tracklet in tracklets:
            tid = tracklet[-1]
            self.tracked_fire_and_smoke[tid].append(tracklet[:-1])
            self.last_appear[tid] = time.time()

    def remove_disappeared_objects(self):
        current_time = time.time()
        for tid in list(self.last_appear.keys()):
            last_time = self.last_appear[tid]

            if (current_time - last_time) > 5: # in second
                del self.last_appear[tid]

                if tid in self.tracked_fire_and_smoke[tid]:
                    del self.tracked_fire_and_smoke[tid]
    
    def __call__(self, batch):
        return self.detect_fire_and_smoke(batch)