# from threading import Thread,Lock
import base64
import os
import shutil
import subprocess
import sys
import threading
import time
from re import compile

# from FindCorner import align_image


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
import tools.infer.predict_cls as predict_cls
import tools.infer.predict_det as predict_det
import tools.infer.predict_rec as predict_rec
import tools.infer.utility as utility
from PIL import Image
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import check_and_read_gif, get_image_file_list
from tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image

logger = get_logger()


lock = threading.Lock()

W_FRAME_OUT = 1260
H_FRAME_OUT = 720
VIEW_H = 3
VIEW_V = 2
W_FRAME_SUB = W_FRAME_OUT / VIEW_H
H_FRAME_SUB = H_FRAME_OUT / VIEW_V


######################
def find_miss_corner(coordinate_dict):
    position_name = ["top-left", "top-right", "bottom-left", "bottom-right"]
    position_index = np.array([0, 0, 0, 0])

    for name in coordinate_dict.keys():
        if name in position_name:
            position_index[position_name.index(name)] = 1

    index = np.argmin(position_index)

    return index


def calculate_missed_coord_corner(coordinate_dict):
    thresh = 0

    index = find_miss_corner(coordinate_dict)

    # calculate missed corner coordinate
    # case 1: missed corner is "top_left"
    if index == 0:
        midpoint = (
            np.add(coordinate_dict["top-right"], coordinate_dict["bottom-left"]) / 2
        )
        y = 2 * midpoint[1] - coordinate_dict["bottom-right"][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict["bottom-right"][0] - thresh
        coordinate_dict["top-left"] = (x, y)
    elif index == 1:  # "top_right"
        midpoint = (
            np.add(coordinate_dict["top-left"], coordinate_dict["bottom-right"]) / 2
        )
        y = 2 * midpoint[1] - coordinate_dict["bottom-left"][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict["bottom-left"][0] - thresh
        coordinate_dict["top-right"] = (x, y)
    elif index == 2:  # "bottom_left"
        midpoint = (
            np.add(coordinate_dict["top-left"], coordinate_dict["bottom-right"]) / 2
        )
        y = 2 * midpoint[1] - coordinate_dict["top-right"][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict["top-right"][0] - thresh
        coordinate_dict["bottom-left"] = (x, y)
    elif index == 3:  # "bottom_right"
        midpoint = (
            np.add(coordinate_dict["bottom-left"], coordinate_dict["top-right"]) / 2
        )
        y = 2 * midpoint[1] - coordinate_dict["top-left"][1] - thresh
        x = 2 * midpoint[0] - coordinate_dict["top-left"][0] - thresh
        coordinate_dict["bottom-right"] = (x, y)

    # print(coordinate_dict)

    return coordinate_dict


def perspective_transform(
    image, source_points, type_lp, width, height
):  # type_lp = 1 -> biển dài, type_lp = 0 -> biển ngắn
    # if type_lp == 1:
    width = int(width)
    height = int(height)
    dest_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, M, (width, height))
    # else:
    #     dest_points = np.float32([[0, 0], [500, 0], [500, 300], [0, 300]])
    #     M = cv2.getPerspectiveTransform(source_points, dest_points)
    #     dst = cv2.warpPerspective(image, M, (500, 300))

    return dst


def align_image(image, coordinate_dict):
    """
    input:
        image: image plate croped frome image src
        coordinate_dict: coordinate_dict top-right' 'bottom-right' 'top-left' 'bottom-left'
    """
    if len(coordinate_dict) < 3:
        # raise ValueError('Please try again')
        return False

    # convert (xmin, ymin, xmax, ymax) to (x_center, y_center)
    # coordinate_dict = get_center_point(coordinate_dict)

    # print(f"length coordinate dict: {len(coordinate_dict)}")
    if len(coordinate_dict) == 3:
        coordinate_dict = calculate_missed_coord_corner(coordinate_dict)

    top_left_point = coordinate_dict["top-left"]
    top_right_point = coordinate_dict["top-right"]
    bottom_right_point = coordinate_dict["bottom-right"]
    bottom_left_point = coordinate_dict["bottom-left"]

    source_points = np.float32(
        [top_left_point, top_right_point, bottom_right_point, bottom_left_point]
    )

    width = np.linalg.norm(
        np.array(
            [
                top_left_point[0] - top_right_point[0],
                top_left_point[1] - top_right_point[1],
            ]
        )
    )
    height = np.linalg.norm(
        np.array(
            [
                top_left_point[0] - bottom_left_point[0],
                top_left_point[1] - bottom_left_point[1],
            ]
        )
    )
    # width, height = wb, hb
    # print(f"width: {width}, height: {height}")
    if width > 2.5 * height:
        type_lp = 1
    else:
        type_lp = 0
    point_type = np.float32([(0, 0), (width, 0), (width, height), (0, height)])
    # transform image and crop
    crop = perspective_transform(image, source_points, point_type, width, height)

    return crop


#####################


# check plate sqare:
def check_plate_sqare(img_plate):
    """
    if plate sqare : Split the plate in half then merge it into 1 line
    if not plate square: return 0
    """
    output_dir = "output_images_split"
    height, width, _ = img_plate.shape
    scale = int(width / height)
    img_list = []
    if scale < 2:
        x3, y3, x4, y4 = 0, int(height / 2), width, height
        x1, y1, x2, y2 = 0, 0, width, height - int(height / 2)
        up_plate = img_plate[y1:y2, x1:x2]
        down_plate = img_plate[y3:y4, x3:x4]
        down_plate = cv2.resize(down_plate, (up_plate.shape[1], up_plate.shape[0]))
        horizontal_plate = cv2.hconcat([up_plate, down_plate])
        img_list = [up_plate, down_plate]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for i in range(1, 21):
            cv2.imwrite(os.path.join(output_dir,f'top_{i}.jpg'), up_plate)
            cv2.imwrite(os.path.join(output_dir,f'bot_{i}.jpg'), down_plate)
        # cv2.waitKey(0)
        # print("test_square", up_plate)
        return horizontal_plate, img_list
    else:
        return None, None


def mode_sys(dt_boxes, rec_res, img, args, count_plate):
    font_path = args.vis_font_path
    drop_score = args.drop_score
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    boxes = dt_boxes
    txts = [rec_res[i][0] for i in range(len(rec_res))]
    scores = [rec_res[i][1] for i in range(len(rec_res))]
    draw_img = draw_ocr_box_txt(
        image, boxes, txts, scores, drop_score=drop_score, font_path=font_path
    )
    save_plate = "save_img_crop"
    os.makedirs(save_plate, exist_ok=True)
    cv2.imwrite(
        os.path.join(save_plate, str(count_plate) + ".jpg"), draw_img[:, :, ::-1]
    )


def mode_rec(rec_res, threshold):
    # print('rec_res[0] : ', rec_res)
    check_acc = False
    txt_result = ""
    sum_acc = 0
    count_txt = 0
    for txt in rec_res:
        # print('------------------------------',str(txt[0])) #("廖'纳绚", 0.9587153196334839)
        acc = round(txt[1], 4)
        print('acc',acc)
        if acc < threshold:
            check_acc = True
        txt_result += "" + str(txt[0])
        sum_acc += acc
        count_txt += 1
    arv_acc = sum_acc / count_txt
    return txt_result, check_acc, arv_acc


def mode_det():
    pass


# def check_format_plate(lp):
#     """
#     lp: text plate
#     return: True or False
#     """
#     plate_format1 = compile("^[0-9]{2}[A-Z]{1}[0-9]{4}$")
#     plate_format2 = compile("^[0-9]{2}[A-Z]{1}[0-9]{5}$")
#     plate_format3 = compile("^[0-9]{2}[A-Z]{1}[0-9]{6}$")
#     plate_format4 = compile("^[0-9]{2}[A-Z]{1}[0-9]{3}\.[0-9]{2}$")
#     return (
#         plate_format1.match(lp) is not None
#         or plate_format2.match(lp) is not None
#         or plate_format3.match(lp) is not None
#         or plate_format4.match(lp) is not None

#     )


def check_format_plate_append(lp):
    """
    lp: text plate
    return: True or False
    """
    plate_format1 = compile("^[0-9]{2}[A-Z]{1}[0-9]{4}$")
    plate_format2 = compile("^[0-9]{2}[A-Z]{1}[0-9]{5}$")
    plate_format3 = compile("^[0-9]{2}[A-Z]{1}[0-9]{6}$")
    plate_format4 = compile("^[0-9]{2}[A-Z]{1}[0-9]{3}[0-9]{2}$")
    if plate_format1.match(lp) is not None:
        lp += "##"
        return lp
    if plate_format2.match(lp) is not None:
        lp += "#"
        return lp
    if plate_format4.match(lp) is not None:
        lp += "#"
        return lp
    if plate_format3.match(lp) is not None:
        return lp
    
    return None

def check_format_plate(lp):
    """
    lp: text plate
    return: True or False
    """
    plate_format_regular  = compile("^[0-9]{2}[ABCDEFGHIJKLMNOPQRSTUVWXYZĐ]{1,2}([0-9]{4,5}|[0-9]{6})$")
    plate_format_special  = compile("^[0-9]{4,5}[A-Z]{2}[0-9]{2,3}$")
    plate_format_temporal = compile("^[A-Z]{0,1}[0-9]{6,7}")

    check = (plate_format_regular.match(lp) is not None) | \
            (plate_format_special.match(lp) is not None) | \
            (plate_format_temporal.match(lp) is not None)

    return check


def show_multi_cam(buffer_frames, buffer_imgs):
    out_img = np.zeros((H_FRAME_OUT, W_FRAME_OUT, 3), np.uint8)
    img_location = [(0, 0), (420, 0), (840, 0), (0, 360), (420, 360), (840, 360)]
    for index in buffer_imgs:
        if len(buffer_imgs[str(index)]) != 0:
            x1 = img_location[int(index)][0]
            y1 = img_location[int(index)][1]
            x2 = int(x1 + W_FRAME_SUB)
            y2 = int(y1 + H_FRAME_SUB)
            # print(x1,y1,x2,y2)
            resized_image = cv2.resize(
                buffer_imgs[str(index)], (int(W_FRAME_SUB), int(H_FRAME_SUB))
            )
            out_img[y1:y2, x1:x2] = resized_image
    return out_img


def Check_id(id, DICT_IDS, DICT_IDS_REMOVE, DICT_IDS_COUNT_APPEAR):
    """
    DICT_IDS
    id

    check_id: True/False
    return DICT_IDS, check_id
    """
    # filer id plate if id>10 frame => delete

    # check have id in DICT_IDS_REMOVE
    print("==============check_id==================")
    print("start")
    print("id: ", id)
    print("DICT_IDS: ", DICT_IDS)
    print("DICT_IDS_REMOVE: ", DICT_IDS_REMOVE)
    print("DICT_IDS_COUNT_APPEAR: ", DICT_IDS_COUNT_APPEAR)
    check_id = True
    if id in DICT_IDS_REMOVE:
        print("id in dict remove")
        return check_id, DICT_IDS, DICT_IDS_REMOVE, DICT_IDS_COUNT_APPEAR

    if id not in DICT_IDS:  # check DICT_IDS empty
        DICT_IDS.update({id: 0})
        DICT_IDS_COUNT_APPEAR.update({id: 0})
    else:
        DICT_IDS[id] += 1
        if DICT_IDS[id] == 9:
            DICT_IDS_REMOVE.append(id)
            del DICT_IDS_COUNT_APPEAR[id]
            del DICT_IDS[id]

    for key in DICT_IDS_COUNT_APPEAR:
        if key == id:
            DICT_IDS_COUNT_APPEAR[key] = 15
        else:
            DICT_IDS_COUNT_APPEAR[key] -= 1
            if DICT_IDS_COUNT_APPEAR[key] == 0:
                DICT_IDS_REMOVE.append(key)
                del DICT_IDS_COUNT_APPEAR[key]
                del DICT_IDS[key]

    # DICT_IDS_REMOVE a must not exceed 1000 elements
    if len(DICT_IDS_REMOVE) > 1000:
        del DICT_IDS_REMOVE[:100]

    check_id = False
    print("end")
    print("id: ", id)
    print("DICT_IDS: ", DICT_IDS)
    print("DICT_IDS_REMOVE: ", DICT_IDS_REMOVE)
    print("DICT_IDS_COUNT_APPEAR: ", DICT_IDS_COUNT_APPEAR)
    return check_id, DICT_IDS, DICT_IDS_REMOVE, DICT_IDS_COUNT_APPEAR
