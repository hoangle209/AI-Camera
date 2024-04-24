
import sys
sys.path.insert(1, ".")

import numpy as np

from AI_Camera.speed_estimation.main import SpeedEstimation
from AI_Camera.license_plate.main import LicensePlatePoseDetection
from AI_Camera.coco_det.main import COCODetection
from AI_Camera.abnormal.main import Abnormal
from AI_Camera.fire_and_smoke_detection.main import FireSmokeDetection
from AI_Camera.ai_lost.main import AILost


if __name__ == "__main__":
    speed_est = SpeedEstimation("configs/yolov8_bytetrack.yaml")
    license_plate = LicensePlatePoseDetection("configs/license_plate_yolov8_pose.yaml")
    coco_det = COCODetection("configs/default.yaml")
    ab = Abnormal("configs/abnormal_cls.yaml")
    fire_and_smoke = FireSmokeDetection("configs/fire_and_smoke_yolov5_det.yaml")

    # AI Lost dang lam nhe

    dummy = np.random.rand(1080, 1920, 3)

    license_plate_ret = license_plate([dummy])
    coco_ret = coco_det([dummy])
    fs_ret = fire_and_smoke([dummy])

    while True:
        ab_ret = ab(dummy)
        speed_ret = speed_est(dummy)
