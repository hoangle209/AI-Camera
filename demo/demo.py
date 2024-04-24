
import sys
sys.path.insert(1, ".")

import cv2 
import numpy as np


from AI_Camera.speed_estimation.main import SpeedEstimation
# from AI_Camera.license_plate.main import LicensePlatePoseDetection
from AI_Camera.coco_det.main import COCODetection
from AI_Camera.abnormal.main import Abnormal
from AI_Camera.fire_and_smoke_detection.main import FireSmokeDetection
from AI_Camera.ai_lost.main import AILost

POINTS = [(1400, 249),
          (990, 267),
          (857, 608),
          (656, 1261),
          (1525, 1254)]

POINTS = [(364, 244), (248, 477), (921,544), (971,329)]

if __name__ == "__main__":
    ai_lost = AILost("configs/ai_lost_items.yaml")
    # coco = COCODetection("configs/ai_lost_items.yaml")
    ai_lost.setup_management_region(POINTS)
    cap = cv2.VideoCapture("192.168.6.254_ch3_20240424172551_20240424173407.mp4")
    
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("frame", 1000, 700) 
    points = np.array(POINTS).astype(np.int32)

    while True:
        suc, frame = cap.read()
        if not suc:
            break
        cv2.polylines(frame, [points], True, (0, 255, 0), 2)
        
        # dets = ai_lost.detector.do_detect(frame)[0]
        # dets = ai_lost.tracker.do_track(dets=dets)
        # for det in dets:
        #     x1, y1, x2, y2 = det[:4]
        #     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        #     cv2.putText(frame, f'{det[-1]}-{det[-2]}', (int(x1), int(y1+10)), cv2.FONT_HERSHEY_SIMPLEX,  
        #                 1, (0,255, 0), 2, cv2.LINE_AA)

        rets = ai_lost.run(frame)
        for tid in rets:
            x1, y1, x2, y2 = rets[tid]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)

        # ret = coco(frame)[0]
        # print(ret)
    
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xff == 27:
            break
    
    cv2.destroyAllWindows()

