import sys
sys.path.insert(1, ".")

import cv2 
import numpy as np


from AI_Camera.speed_estimation.main import SpeedEstimation
from AI_Camera.license_plate.main import LicensePlatePoseDetection
from AI_Camera.coco_det.main import COCODetection
from AI_Camera.abnormal.main import Abnormal
from AI_Camera.fire_and_smoke_detection.main import FireSmokeDetection
from AI_Camera.ai_lost.main import AILost

if __name__ == "__main__":
    cap = cv2.VideoCapture("192.168.6.254_ch2_20240504164354_20240504170023.mp4")
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("frame", 1000, 700) 

    POINTS = [(364, 244), (248, 477), (921,544), (971,329)]
    points = np.array(POINTS).astype(np.int32)
    ai_lost = AILost("configs/ai_lost_items.yaml")
    ai_lost.setup_management_region(POINTS)

    lp = LicensePlatePoseDetection("configs/license_plate_yolov8_pose.yaml")

    while True:
        suc, frame = cap.read()
        if not suc:
            break
        ######################## AI Lost
        # cv2.polylines(frame, [points], True, (0, 255, 0), 2)
    
        # rets = ai_lost.run(frame)
        # if len(rets) > 0:
        #     for tid in rets:
        #         x1, y1, x2, y2 = rets[tid]
        #         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)

        rets = lp.detector.do_detect([frame])
        rets = lp.post_process_kpts(rets)
        for ret in rets[0]:
            x1, y1, x2, y2 = ret[0]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)
        
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xff == 27:
            break
    
    cv2.destroyAllWindows()