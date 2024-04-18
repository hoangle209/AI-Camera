from omegaconf import DictConfig
from third_parties.track.byte_track.byte_tracker import BYTETracker
from AI_Camera.utils import get_pylogger
logger = get_pylogger()

from .model_zoo import tracker_zoo

class Trackers:
    def __init__(self, config) -> None:
        self.load_configurations(config)

    def load_configurations(self, config: DictConfig):
        self.cfg = config

        model = self.cfg.model
        if model == "byte_track":
            self.tracker = BYTETracker(
                            track_thresh=config.kwargs.track_thresh,
                            track_buffer=config.kwargs.track_buffer,
                            match_thresh=config.kwargs.match_thresh,
                            mot20=config.kwargs.mot20,
                            frame_rate=config.kwargs.frame_rate
                        )
            self.kwargs_ = {}
        else:
            logger.error(f"Model type {model} is not supported !!!")
            raise

        self.__track = tracker_zoo[model]
    
    def do_track(
            self,
            dets=None, 
            xyxy=None,
            confidence=None,
            class_id=None
        ):
        tracklets = self.__track(
                        self.tracker,
                        dets, 
                        xyxy,
                        confidence,
                        class_id, 
                        **self.kwargs_
                    ) # xyxy, score, cls_id, track_id, (N, 7)

        return tracklets
        
