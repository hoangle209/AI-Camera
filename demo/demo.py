
import sys
sys.path.insert(1, ".")

import cv2 
import matplotlib.pyplot as plt 
from AI_Camera.speed_estimation.main import SpeedEstimation
# from AI_Camera.license_plate.main import LicensePlatePoseDetection
from AI_Camera.coco_det.main import COCODetection
from AI_Camera.abnormal.main import Abnormal
from AI_Camera.fire_and_smoke_detection.main import FireSmokeDetection
from AI_Camera.ai_lost.main import AILost

if __name__ == "__main__":
    ai_lost = AILost("configs/ai_lost_items.yaml")
    cap = cv2.VideoCapture("rtsp://admin:admin123@192.168.6.203:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif")
    writer = cv2.VideoWriter("test.avi", 
                             cv2.VideoWriter_fourcc(*'MJPG'), 
                             30, (1920,1080))

    while True:
        suc, frame = cap.read()
        if not suc:
            break
        
        dets = ai_lost(frame)[0]
        


        # cv2.imshow("f", frame)
        # if cv2.waitKey(1) & 0xff == 27:
        #     break
    
    cv2.destroyAllWindows()

