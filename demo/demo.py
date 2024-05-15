
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
    coco = COCODetection("configs/default.yaml")
    ai_lost.setup_management_region(POINTS)
    
    speed = SpeedEstimation("configs/yolov8_bytetrack.yaml")

    cap = cv2.VideoCapture("video/vlc-record-2024-05-06-13h40m59s-192.168.6.254_ch2_20240504164354_20240504170023.mp4-.mp4")
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
   
    size = (frame_width, frame_height)
    
    # cv2.namedWindow("frame", cv2.WINDOW_NORMAL) 
    # cv2.resizeWindow("frame", 1000, 700) 
    points = np.array(POINTS).astype(np.int32)

    writer = cv2.VideoWriter("out.avi", cv2.VideoWriter_fourcc(*'MJPG'),
                             30, size)

    c = 0
    try:
        while True:
            suc, frame = cap.read()
            if not suc:
                break
            # cv2.polylines(frame, [points], True, (0, 255, 0), 2)
            
            _, _, rets = speed(frame)

            if len(rets) > 0:
                for tid in rets:
                    # print(tid, rets[tid])
                    # print("######################################")

                    det, s = rets[tid]
                    # det[det<0] = 0 
                    (x1, y1, x2, y2) = det
                    cv2.rectangle(frame, (int(x1),int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                    cv2.putText(frame, f"{s}", (int(x1), int(y2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)

                writer.write(frame)
            
            c+=1
    except KeyboardInterrupt:
        print('Stopped by keyboard interrupt')
    writer.release()

            
        #     print(rets)
 

        # rets = ai_lost.run(frame)
        # if len(rets) > 0:
        #     for tid in rets:
        #         x1, y1, x2, y2 = rets[tid]
        #         cv2.rectangle(frame, (x1,y1), (x2, y2), (255, 0, 0), 4)
        #     cv2.imwrite(f"results/{c}.jpg", frame)
        #     print(rets)
        # print(c)
        # c+=1

        # ret = coco(frame)[0]
        
    
        # cv2.imshow("frame", frame)
        # if cv2.waitKey(1) & 0xff == 27:
        #     break
    
    # cv2.destroyAllWindows()

