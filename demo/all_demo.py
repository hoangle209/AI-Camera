
import sys
sys.path.insert(1, ".")

import numpy as np

from AI_Camera.speed_estimation.main import SpeedEstimation
# from AI_Camera.license_plate.main import LicensePlatePoseDetection
from AI_Camera.coco_det.main import COCODetection
from AI_Camera.abnormal.main import Abnormal
from AI_Camera.fire_and_smoke_detection.main import FireSmokeDetection
from AI_Camera.ai_lost.main import AILost

POINTS = [(364, 244), (248, 477), (921,544), (971,329)]

if __name__ == "__main__":
    speed_est = SpeedEstimation("configs/yolov8_bytetrack.yaml")
    # license_plate = LicensePlatePoseDetection("configs/license_plate_yolov8_pose.yaml")
    coco_det = COCODetection("configs/default.yaml")
    ab = Abnormal("configs/abnormal_cls.yaml")
    fire_and_smoke = FireSmokeDetection("configs/fire_and_smoke_yolov5_det.yaml")

    ai_lost = AILost("configs/ai_lost_items.yaml")
    ai_lost.setup_management_region(POINTS)

    dummy = np.random.rand(1080, 1920, 3)

    # license_plate_ret = license_plate([dummy]) # List [ list [id, txt_result, [x1, y1, x2, y2], [], [], []] ]
    coco_ret = coco_det([dummy]) # List [ list cac box trong 1 frame ]
    # fs_ret = fire_and_smoke([dummy]) # List [ list cac box trong 1 frame ]

    # while True:
    #     ab_ret = ab(dummy) # True hoac False 
    #     speed_ret = speed_est(dummy) # list cac box | list string toc do | {tid: box, speed} 
    #     lost = ai_lost.run(dummy) # {tid: box}


