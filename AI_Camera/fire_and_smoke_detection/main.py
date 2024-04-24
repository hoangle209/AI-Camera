from AI_Camera.core.main import BaseModule
from AI_Camera.utils import get_pylogger
logger = get_pylogger()


class FireSmokeDetection(BaseModule):
    def __init__(self, config) -> None:
        super().__init__(config)

    def detect_fire_and_smoke(self, batch):
        dets = self.detector.do_detect(batch)
        return dets
    
    def __call__(self, batch):
        return self.detect_fire_and_smoke(batch)