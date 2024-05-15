from collections import defaultdict, deque
import time
from omegaconf import OmegaConf
import numpy as np
import supervision as sv
from scipy.stats import variation
from third_parties.track.simple_track.sim_tracker import ObjectTracker


from AI_Camera.core.main import BaseModule
from AI_Camera.utils import get_pylogger
logger = get_pylogger()


class FireSmokeDetection(BaseModule):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.tracked_fire_and_smoke = defaultdict(lambda: deque(maxlen=self.cfg.inspector.confirm_thresh))
        self.last_appear = {}
        self.area_overtime = defaultdict(lambda: deque(maxlen=self.cfg.inspector.confirm_thresh // 2))

        if self.cfg.use_slicer:
            kwargs = OmegaConf.to_object(self.cfg.slicer)
            self.slicer = sv.InferenceSlicer(callback=self.slicer_callback, **kwargs)
        
        self.sim_track = ObjectTracker(window_size=self.cfg.inspector.window_size, 
                                       tolerance=self.cfg.inspector.tolerance)


    def slicer_callback(self, batch):
        dets = self.detector.do_detect(batch)[0]
        dets = np.array(dets)
        return sv.Detections(
            xyxy=dets[:, :4],
            confidence=dets[:, 4],
            class_id=dets[:, 5]
        )


    def detect_fire_and_smoke(self, batch):
        if self.cfg.use_slicer:
            dets = []
            for im in batch:
                det = self.slicer(im)
                dets.append(
                    np.concatenate(
                        [det.xyxy, 
                         det.confidence[:, None],
                         det.class_id[:, None]], axis=1
                    )
                )
        else:
            dets = self.detector.do_detect(batch)
        return dets
    

    # def confirm_fire_and_smoke_by_consistance(self, dets):
    #     tracklets = self.tracker.do_track(dets=dets)
    #     outs = []
    #     for tracklet in tracklets:
    #         tid = tracklet[-1]
    #         x1, y1, x2, y2 = tracklet[:4]
    #         self.area_overtime[tid].append((x2-x1)*(y2-y1))
    #         self.last_appear[tid] = time.time()
    #         if len(self.area_overtime[tid]) > 9:
    #             id_area_overtime = np.array(self.area_overtime[tid])
    #             consistance = (np.std(id_area_overtime) / np.mean(id_area_overtime)) < 0.05

    #             if not consistance:
    #                 outs.append(tracklet[:-1])
    #     return outs

    
    def confirm_fire_and_smoke_by_sim_track(self, dets):
        _, areas, bboxes = self.sim_track.tracking(dets)
        outs = []
        for tid in areas:
            self.area_overtime[tid].append(areas[tid])
            id_area_overtime = np.array(self.area_overtime[tid])
            consistance = variation(np.array(id_area_overtime)) < self.cfg.inspector.variation_thresh
            if not consistance:
                outs.append(bboxes[tid])
        
        return outs
        

    # def confirm_fire_and_smoke_by_track(self, dets):
    #     tracklets = self.tracker.do_track(dets=dets)
    #     outs = []
    #     for tracklet in tracklets:
    #         tid = tracklet[-1]
    #         self.tracked_fire_and_smoke[tid].append(tracklet[:-1])
    #         self.last_appear[tid] = time.time()

    #         if len(self.tracked_fire_and_smoke[tid]) == self.cfg.inspector.confirm_thresh:
    #             outs.append(tracklet[:-1])
        
    #     return outs


    def remove_disappeared_objects(self):
        current_time = time.time()
        for tid in list(self.last_appear.keys()):
            last_time = self.last_appear[tid]

            if (current_time - last_time) > 5: # in second
                del self.last_appear[tid]

                if tid in self.tracked_fire_and_smoke[tid]:
                    del self.tracked_fire_and_smoke[tid]
    

    def run(self, batch):
        dets = self.detect_fire_and_smoke(batch)[0]
        confirm = self.confirm_fire_and_smoke(dets)
        self.remove_disappeared_objects()

        return confirm
    
    
    def __call__(self, batch):
        return self.detect_fire_and_smoke(batch)