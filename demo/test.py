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

if __name__ == "__main__":
    cap = cv2.VideoCapture("videos\\lost2.mp4")
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("frame", 1000, 700) 

    # POINTS = [(364, 244), (248, 477), (921,544), (971,329)]
    POINTS = [(364, 244), (248, 777), (921,844), (971,329)]
    points = np.array(POINTS).astype(np.int32)
    ai_lost = AILost("configs/ai_lost_items.yaml")
    ai_lost.setup_management_region(POINTS)

    # lp = LicensePlatePoseDetection("configs/license_plate_yolov8_pose.yaml")

    # fs_model = FireSmokeDetection("configs/fire_and_smoke_yolov5_det.yaml")
    # print(fs_model.detector.model.names)
    # frame = cv2.imread("videos\\imgpsh_fullsize_anim (2).jpg")
    c = 0
    while True:
        suc, frame = cap.read()
        if not suc:
            break
        ####################### AI Lost
        cv2.polylines(frame, [points], True, (0, 255, 0), 2)

        # dets = ai_lost.detector.do_detect([frame])[0]
        # dets = ai_lost.tracker.do_track(dets)
        # for det in dets:
        #     x1, y1, x2, y2, conf, cls_id, tid = det
        #     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        #     cv2.putText(frame, f"{tid}-{conf:.2f}", (int(x1), int(y2)),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)

        rets = ai_lost.run(frame)
        if len(rets) > 0:
            for tid in rets:
                x1, y1, x2, y2 = rets[tid]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)

        # dets = fs_model.detect_fire_and_smoke([frame])[0]
        
        # for det in dets:
        #     x1, y1, x2, y2, conf, cls_id = det
        #     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)
        #     cv2.putText(frame, f"{cls_id}-{conf:.2f}", (int(x1), int(y2)),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
        # if len(dets) > 0:
        #     cv2.imwrite(f"results\\{c}.jpg", frame)
        c+=1
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xff == 27:
            break
    
    cv2.destroyAllWindows()