
import sys
sys.path.insert(1, ".")

import cv2 
import numpy as np
import math
from AI_Camera.fire_and_smoke_detection.main import FireSmokeDetection

def griding(image, imsz=(1080,1920), insz=(640, 640)):
    imh, imw = imsz
    inh, inw = insz
    num_h = math.ceil(imh/inh)
    num_w = math.ceil(imw/inw)

    pad_im = np.zeros((int(inh*num_h), int(inw*num_w), 3))
    pad_im[:imh, :imw] = image

    batch = []
    for i in range(num_w):
        for j in range(num_h):
            batch.append(
                pad_im[int(j*inh):int((j+1)*inh),
                       int(i*inw):int((i+1)*inw), :]
            )
    return batch, (num_h, num_w)


def remap(rets, num, insz=(640,640)):
    inh, inw = insz
    numh, numw = num

    final = []
    for i, ret in enumerate(rets):
        cw = i // numh
        ch = i % numh
        for det in ret:
            x1,y1,x2,y2,conf,cls_id = det
            x1 += cw*inw
            x2 += cw*inw
            y1 += ch*inh
            y2 += ch*inh

            final.append((x1,y1,x2,y2,conf,cls_id))
    return final


if __name__ == "__main__":
    model = FireSmokeDetection("configs/fire_and_smoke_yolov5_det.yaml")

    cap = cv2.VideoCapture("videos\\fire1.mp4")

    # frame_width = int(cap.get(3)) 
    # frame_height = int(cap.get(4)) 
    # size = (frame_width, frame_height)
    # writer = cv2.VideoWriter("out.avi", cv2.VideoWriter_fourcc(*'MJPG'),
    #                          30, size)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("frame", 1000, 700) 

    c = 0
    try:
        while True:
            suc, frame = cap.read()
            if not suc:
                break

            dets = model([frame])[0]
            # print("first round: ", dets)
            dets = model.confirm_fire_and_smoke_by_sim_track(dets)
            # print("second round: ", dets)
            # print("-------------------------------------------------------")


            if len(dets) > 0:
                for det in dets:
                    x1, y1, x2, y2, conf, cls_id = det
                    cv2.rectangle(frame, (int(x1),int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, f"{cls_id}-{conf:.2f}", (int(x1), int(y2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)

                # writer.write(frame)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xff == 27:
                break
            
            c += 1

    except KeyboardInterrupt:
        print('Stopped by keyboard interrupt')
    
    # writer.release()
    cv2.destroyAllWindows()

    # im = cv2.imread("videos\\imgpsh_fullsize_anim.jpg")
    # im = cv2.resize(im, (1920,900))
    # cv2.rectangle(im, (780,300), (820,340), (0,255,0), 1)
    # cv2.imshow("frame", im)
    # cv2.waitKey(0)

