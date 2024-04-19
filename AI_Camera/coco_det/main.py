from AI_Camera.core.main import BaseModule
from AI_Camera.utils import get_pylogger
logger = get_pylogger()


class COCODetection(BaseModule):
    def __init__(self, config) -> None:
        super().__init__(config)

    def detect_coco(self, batch):
        dets = self.detector.do_detect(batch)
        return dets
    
    def __call__(self, batch):
        return self.detect_coco(batch)