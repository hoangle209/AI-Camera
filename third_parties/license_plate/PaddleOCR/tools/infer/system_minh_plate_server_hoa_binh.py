# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import os
import shutil
import subprocess
import sys
import threading

####inclue
from include_align import *
from predict_rec import TextRecognizer
from predict_system import TextSystem

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import copy
import json
import logging
import time

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import tools.infer.predict_cls as predict_cls
import tools.infer.predict_det as predict_det
import tools.infer.predict_rec as predict_rec
import tools.infer.utility as utility
from PIL import Image
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import check_and_read_gif, get_image_file_list
from tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image

logger = get_logger()


# The callback for when the client connects to the broker.
def on_connect(client, userdata, flags, rc):
    print("Connected To Broker")
    # After establishing a connection, subscribe to the input topic.
    client.subscribe("test")


# The callback for when a message is received from the broker.
def on_message(client, userdata, msg):
    # Decode the message payload from Bytes to String.
    global DATA_MQTT
    global CHECK_TERMINATE
    global LEN_BUFFER_IMG
    payload = msg.payload.decode("UTF-8")
    # Print the payload to the console.
    # print('check on_message')
    if payload == "quit":
        # print('quit in if 1 ')
        # If `quit` disconnect the client, ending the program.
        client.disconnect()
        CHECK_TERMINATE = True
    else:
        # print('receive data')
        res = json.loads(payload)
        lock.acquire()
        if len(DATA_MQTT) < LEN_BUFFER_IMG:
            DATA_MQTT.append(res)
            print("len buffer receive :", len(DATA_MQTT))
        lock.release()


def mqtt_fuc():
    print("========================threading 1=================================")
    # Define an Id for the client to use.
    Id = "consumerPy"
    # Define the Ip address of the broker.
    Ip = "192.168.6.33"
    # Create a client.
    client = mqtt.Client(Id)
    # Set the callback functions of the client for connecting and incoming messages.
    client.on_connect = on_connect
    client.on_message = on_message
    # Then, connect to the broker.
    port = 1883
    keepakive = 60
    # print('check')
    client.connect(Ip, 1883, 120)
    # Finally, process messages until a `client.disconnect()` is called.
    # print('check')
    client.loop_forever()


def main(args):
    print("========================threading 2=================================")
    global DATA_MQTT
    global COUNT_IMG
    global CHECK_TERMINATE
    global MODE_REC
    global MODE_SYS
    global MODE_DET
    global BUFFER_IMGS
    global IS_VISUALIZE
    global SAVE_VIDEO
    global DICT_IDS
    global DICT_IDS_REMOVE
    global DICT_IDS_COUNT_APPEAR
    threshold = 0.94  # check reg plate smaller then exit
    #############load model paddle#############
    if MODE_SYS:
        print("run MODE_SYS")
        text_sys = TextSystem(args)
    if MODE_REC:
        print("run MODE_REC")
        text_recognizer = TextRecognizer(args)
        # 3, 48, 320
        hight_ini = 48
        weigh_ini = 320
        img_ini = out_img = np.zeros((hight_ini, weigh_ini, 3), np.uint8)
        rec_res_ini, _ = text_recognizer([img_ini])
        print("run MODE_REC done")
    if MODE_DET:
        print("run MODE_DET")
        text_detector = TextDetector(args)
    print("load model done 1")
    # is_visualize = True
    font_path = args.vis_font_path
    drop_score = args.drop_score
    draw_img_save_dir = args.draw_img_save_dir

    save_results = []
    total_time = 0
    cpu_mem, gpu_mem, gpu_util = 0, 0, 0
    _st = time.time()
    count = 0
    print("load model done 2")
    ##########################################

    while True:
        if len(DATA_MQTT) != 0:
            lock.acquire()
            res = DATA_MQTT.pop(0)
            print("len buffer pop :", len(DATA_MQTT))
            lock.release()
            time.sleep(1)
            # print('iiiiiiiii')
            data = base64.b64decode(res["image"])
            box_kps_str = res["id_box_kps"]
            camera_id = res["camera_id"]

            # check_comtinue_reg_flate

            # buffer_frame = res["buffer_frame"]13162
            image_decode = cv2.imdecode(
                np.fromstring(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED
            )

            if COUNT_IMG == 0 and SAVE_VIDEO:
                frame_height, frame_width = image_decode.shape[0], image_decode.shape[1]
                frame_width, frame_height = 1260, 720
                out = cv2.VideoWriter(
                    "outpy.avi",
                    cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                    10,
                    (frame_width, frame_height),
                )

            if box_kps_str != "":
                box_kpss = box_kps_str.split(",")
                # print('box_kpss : ', box_kpss)
                count_plate = 0
                for box_kps in box_kpss:
                    id = box_kps.split(" ")[0]

                    # check_id, DICT_IDS, DICT_IDS_REMOVE, DICT_IDS_COUNT_APPEAR = Check_id(id, DICT_IDS, DICT_IDS_REMOVE, DICT_IDS_COUNT_APPEAR)
                    # if check_id:
                    #     print('done check_id')
                    #     continue
                    # else:
                    #     continue

                    xyxy = box_kps.split(" ")[1:5]
                    kps_string = box_kps.split(" ")[5:]
                    kps_int = [int(float(kp)) for kp in kps_string]
                    # print('kps_int : ', kps_int)
                    kps = [
                        (kps_int[0], kps_int[1]),
                        (kps_int[2], kps_int[3]),
                        (kps_int[4], kps_int[5]),
                        (kps_int[6], kps_int[7]),
                        (kps_int[8], kps_int[9]),
                    ]
                    # print('kps : ', kps)
                    x1, y1, x2, y2 = (
                        int(float(xyxy[0])),
                        int(float(xyxy[1])),
                        int(float(xyxy[2])),
                        int(float(xyxy[3])),
                    )

                    # count_kp = 0
                    coordinate_dict = {
                        "top-right": (int(kps[1][0] - x1), int(kps[1][1] - y1)),
                        "bottom-right": (int(kps[4][0] - x1), int(kps[4][1] - y1)),
                        "top-left": (int(kps[0][0] - x1), int(kps[0][1] - y1)),
                        "bottom-left": (int(kps[3][0] - x1), int(kps[3][1] - y1)),
                    }

                    img_crop_lp = image_decode[y1:y2, x1:x2]
                    img_copy = img_crop_lp.copy()
                    cv2.imwrite("img_crop_lp_old.jpg", img_copy)
                    # align image
                    cropped_img = align_image(img_copy, coordinate_dict)
                    cv2.imwrite("cropped_img_new.jpg", cropped_img)

                    ####################detect and recognize#############################
                    img_list = []
                    # detect and recognize in horizontal number plate
                    check_plate_sqare_, img_list = check_plate_sqare(cropped_img)
                    if check_plate_sqare_ is not None:
                        img = check_plate_sqare_
                        cv2.imwrite("check_plate_sqare_0.jpg", img_list[0])
                        cv2.imwrite("check_plate_sqare_1.jpg", img_list[1])
                    else:
                        img = cropped_img
                        img_list = [cropped_img]
                    starttime = time.time()
                    if (
                        MODE_SYS
                    ):  # If you combine the plate into 1 line, it will be considered as 1 word sometime 2 word
                        dt_boxes, rec_res = text_sys(img)
                        txt = ""
                        for text, score in rec_res:
                            txt = str(text) + ":" + str(round(score, 2)) + "--"
                        if txt == "":
                            txt = "Null"
                        image_decode = cv2.putText(
                            image_decode,
                            txt,
                            (x1 - 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        # save img crop
                        mode_sys(dt_boxes, rec_res, img, args, count_plate)
                    if MODE_REC:
                        print("wait text_recognizer")
                        rec_res, _ = text_recognizer(img_list)
                        txt_result, check_acc, arv_acc = mode_rec(rec_res, threshold)
                        # if check_acc: #samller threshold
                        #     continue

                        result_format_check = check_format_plate(txt_result)
                        if result_format_check == False:
                            print("CHECK FORMAT PLATE FALSE")
                        txt_result += " " + str(round(arv_acc, 4))
                        image_decode = cv2.putText(
                            image_decode,
                            txt_result,
                            (x1 - 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 255, 255),
                            3,
                            cv2.LINE_AA,
                        )
                    if MODE_DET:
                        dt_boxes, _ = text_detector(img)

                    elapse = time.time() - starttime
                    print("time inferen det and rec : ", elapse)
                    #####################################################################
                    count_plate += 1
                    if IS_VISUALIZE or SAVE_VIDEO:
                        print("IS_VISUALIZE or SAVE_VIDEO")
                        cv2.rectangle(image_decode, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        for kp in kps:
                            cv2.circle(image_decode, kp, 9, (255, 255, 0), -1)
                BUFFER_IMGS[str(camera_id)] = image_decode
                out_img = show_multi_cam(6, BUFFER_IMGS)
                if IS_VISUALIZE:
                    print("===================check show==============================")
                    # BUFFER_IMGS[str(camera_id)] = image_decode
                    # out_img = show_multi_cam(6, BUFFER_IMGS)
                    # print(out_img)
                    cv2.imshow("show", out_img)
                    print("done show")
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        CHECK_TERMINATE = True
                        DATA_MQTT = []
                        break
                    print("done waikey")
                if SAVE_VIDEO:
                    out.write(out_img)
                COUNT_IMG += 1
        if CHECK_TERMINATE and len(DATA_MQTT) == 0:
            print("terminate program")
            print("total time : ", total_time)
            break


if __name__ == "__main__":
    global DATA_MQTT  # DATA_MQTT buffer (list) receive data frem MQTT
    global COUNT_IMG  # count image receive
    global CHECK_TERMINATE  # if retinaface send messege terminate then terminate program
    global MODE_REC  # only rec
    global MODE_SYS  # det and rec
    global MODE_DET  # only det
    global IS_VISUALIZE  # save and show video
    global SAVE_VIDEO
    global BUFFER_IMGS  # BUFFER_IMGS
    global LEN_BUFFER_IMG
    global DICT_IDS
    global DICT_IDS_REMOVE
    global DICT_IDS_COUNT_APPEAR

    CHECK_TERMINATE = False
    IS_VISUALIZE = True
    SAVE_VIDEO = False
    DATA_MQTT = []
    COUNT_IMG = 0
    draw_img_save_dir = "image"
    LEN_BUFFER_IMG = 2
    DICT_IDS = {}  # id: count
    DICT_IDS_REMOVE = []
    DICT_IDS_COUNT_APPEAR = {}

    print("IS_VISUALIZE : ", IS_VISUALIZE)
    print("SAVE_VIDEO : ", SAVE_VIDEO)

    # set up show image
    buffer_frame = 6
    BUFFER_IMGS = {}
    for key_buff in range(0, buffer_frame):
        name_key = str(key_buff)
        BUFFER_IMGS.update({name_key: []})

    # create foler save image
    if os.path.exists(draw_img_save_dir) and os.path.isdir(draw_img_save_dir):
        shutil.rmtree(draw_img_save_dir)
        os.makedirs(draw_img_save_dir)
    os.makedirs(draw_img_save_dir, exist_ok=True)

    ##########################threading 1############################
    thread1 = threading.Thread(name="mqtt_fuc", target=mqtt_fuc)
    thread1.start()

    ##########################threading 2############################
    # config model in utility.py
    MODE_REC = True  # mode recognition
    MODE_DET = False  # mode detect text
    MODE_SYS = False  # mode both recognition and detect text
    args = utility.parse_args()
    if MODE_REC:
        args.rec_model_dir = "./rec_lp_lite_48x160_only_reg_plate"
        args.rec_image_shape = "3, 48, 160"
        args.rec_char_dict_path = "./ppocr/utils/rec_lp_lite_48x160_only_reg_plate.txt"
    main(args)
