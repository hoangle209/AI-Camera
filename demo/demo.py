
import sys
sys.path.insert(1, ".")

import cv2 
from AI_Camera.speed_estimation.main import SpeedEstimation
from AI_Camera.license_plate.main import LicensePlatePoseDetection

if __name__ == "__main__":
    # speed = SpeedEstimation("configs/yolov8_bytetrack.yaml")

    # cap = cv2.VideoCapture("video/video_1681116180_1681116480.mp4")

    # c = 0
    # while True:
    #     suc, frame = cap.read()

    #     if not suc:
    #         break
        
    #     tracklets, labels = speed.estimate_speed(frame)
    #     print(tracklets)
    #     print("- labels", labels)

    im = cv2.imread("gettyimages-1403633842-612x612.jpg")
    lc = LicensePlatePoseDetection("configs/license_plate_yolov8_pose.yaml")
    r = lc.detect_pose_license_plate([im, im])
    print(r)