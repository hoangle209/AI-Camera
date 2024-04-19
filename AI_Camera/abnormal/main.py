from time import time
import numpy as np
import cv2

from AI_Camera.core.main import BaseModule

class OpticalFlow():
    def __init__(self, s_opt=320):
        self.prev = None
        self.s_opt = s_opt
    
    def resize_bbox(self, box, d_size):
        x1, y1, x2, y2 = box
        x_scale = self.s_opt / d_size[1]
        y_scale = self.s_opt / d_size[0]
        x1 = int(np.round(x1 * x_scale))
        y1 = int(np.round(y1 * y_scale))
        x2 = int(np.round(x2 * x_scale))
        y2 = int(np.round(y2 * y_scale))
        return x1, y1, x2, y2 
        
    def get(self, frame):

        frame = cv2.resize(frame, (self.s_opt, self.s_opt))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev is None:
            self.prev = frame

        flow = cv2.calcOpticalFlowFarneback(self.prev, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev = frame
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        return mag


class Abnormal(BaseModule):
    min_person = 1.4
    min_grad = 2
    live = 5
    min_ab_frames = 3

    def __init__(self, config) -> None:
        super().__init__(config)

        self.h_tracks = {}
        self.c_fs = []
        self.otp = OpticalFlow(320)


    def save(self, tid, speed):
        if tid not in self.h_tracks:
            self.h_tracks[tid] = {"value": [], "last":time()}
        self.h_tracks[tid]["value"].append(speed)
        self.h_tracks[tid]["last"] = time()


    def remove_unuse_id(self):
        keys = [k for k in self.h_tracks]
        for k in keys:
            if time() - self.h_tracks[k]["last"] >= self.live:
                del self.h_tracks[k]
    

    def is_abnormal(self):
        c_f = []
        self.remove_unuse_id()

        for k in self.h_tracks:
            val = self.h_tracks[k]["value"]
            # chưa thu thập đủ min_ab_frames
            if len(val) <= self.min_ab_frames:
                continue
            i_get = -self.min_ab_frames if len(val) <= self.min_ab_frames*2 else (-self.min_ab_frames*2)
            grad = np.gradient(val[i_get:])
            grad = np.absolute(grad)
            c_f.append(grad.mean())

        num_person = sum([p >= self.min_grad for p in c_f])
        self.c_fs = self.c_fs[1:] if len(self.c_fs) == self.min_ab_frames else self.c_fs
        self.c_fs.append(num_person)
        return np.mean(self.c_fs) >= self.min_person


    def run_abnormal_det(self, frame):
        human_det = self.detector.do_detect(frame)[0]
        tracklets = self.tracker.do_track(dets=human_det)

        flow = self.otp.get(frame)
        for track in tracklets:
            # update ABNORMAL
            tid = int(track[-1])
            x1, y1, x2, y2 = self.otp.resize_bbox(track[:4], frame.shape[:2])
            speed = flow[y1:y2, x1:x2]
            speed = speed[speed>1]
            speed = 0 if len(speed) <= 0 else speed.mean()

            self.save(tid, speed)
        return self.is_abnormal()
    

    def __call__(self, frame):
        return self.run_abnormal_det(frame)


