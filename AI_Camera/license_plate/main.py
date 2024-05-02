from typing import Any
import numpy as np

from AI_Camera.core.main import BaseModule
from third_parties.license_plate.PaddleOCR.tools.infer.include_align import (
    align_image, 
    check_format_plate, 
    check_format_plate_append, 
    check_plate_sqare, 
    mode_rec
)
from third_parties.license_plate.PaddleOCR.tools.infer.predict_det import TextDetector
from third_parties.license_plate.PaddleOCR.tools.infer.predict_rec import TextRecognizer
from third_parties.license_plate.PaddleOCR.tools.infer.predict_system import TextSystem
import third_parties.license_plate.PaddleOCR.tools.infer.utility as utility

from AI_Camera.utils import get_pylogger
logger = get_pylogger()

class LicensePlatePoseDetection(BaseModule):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.args = utility.parse_args()
        self.MODE_REC = self.cfg.OCR.MODE_REC  # mode recognition
        self.MODE_DET = self.cfg.OCR.MODE_DET  # mode detect text
        self.MODE_SYS = self.cfg.OCR.MODE_SYS  # mode both recognition and detect text

        if self.MODE_REC:
            self.args.rec_model_dir = self.cfg.OCR.rec.rec_model_dir
            self.args.rec_image_shape = self.cfg.OCR.rec.rec_image_shape
            self.args.rec_char_dict_path = self.cfg.OCR.rec.rec_char_dict_path

        self.threshold = self.cfg.OCR.threshold  # check reg plate smaller then exit

        #############load model paddle#############
        if self.MODE_SYS:
            self.text_sys = TextSystem(self.args)

        if self.MODE_REC:
            self.text_recognizer = TextRecognizer(self.args)

        # if self.MODE_DET:
        #     self.text_detector = TextDetector(self.args)
    

    def find_intersection(self, line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        a11 = y1 - y2
        a12 = x2 - x1
        b1 = (x1*y2 - x2*y1)

        a21 = (y3 - y4)
        a22 = (x4 - x3)
        b2 = (x3*y4 - x4*y3)
        A = np.array([[a11, a12],
                      [a21, a22]])

        B = -np.array([b1,b2])

        intersection = np.linalg.solve(A, B)
        x, y = intersection[0], intersection[1]
        
        return x, y  


    def post_process_kpts(self, results):
        boxes, keypoints = results

        plate_points = []
        for batch_keypoint in keypoints:
            plate_point = []
            for keypoint in batch_keypoint:
                if keypoint.shape[0] >= 4:
                    x1, y1 = keypoint[0]
                    x2, y2 = keypoint[1]
                    x3, y3 = keypoint[2]
                    x4, y4 = keypoint[3]
                    xc, yc = self.find_intersection((x1, y1, x3, y3), (x2, y2, x4, y4))
                    plate_point.append([x1, y1, x2, y2, xc, yc, x4, y4, x3, y3])
            
            plate_points.append(plate_point)

        plate_boxes = [boxes_[:, :5].tolist() for boxes_ in boxes]

        dets = [np.concatenate([plate_box, plate_point], axis=1) if len(plate_box) > 0 else [] \
                        for (plate_box, plate_point) in zip(plate_boxes, plate_points)]
        
        results_post = []
        for det in dets:
            result_post = []
            for b in det:
                conf = float(b[4])
                b[b < 0] = 0
                b = b.astype("int")
                b_kps = b[5:]
                result_post.append([b[:4], b_kps, None, conf])
            results_post.append(result_post)

        return results_post
    

    def reg_plate(self, img_raw, box_kpss):
        image_decode = img_raw.copy()
        count_plate = 0
        results = []
        for box_kps in box_kpss:
            id = box_kps[2]
            kps_int = box_kps[1]

            kps = [
                [kps_int[0], kps_int[1]],
                [kps_int[2], kps_int[3]],
                [kps_int[6], kps_int[7]],
                [kps_int[8], kps_int[9]],
                
            ]

            x1, y1, x2, y2 = box_kps[0]
            # count_kp = 0
            bbox = box_kps[0]
            kpt = kps

            pnt0 = np.maximum(kpt[0],bbox[:2])
            pnt1 = np.array([np.minimum(kpt[1][0],bbox[2]), np.maximum(kpt[1][1],bbox[1])])
            pnt2 = np.minimum(kpt[3],bbox[2:4])
            pnt3 = np.array([np.maximum(kpt[2][0],bbox[0]), np.minimum(kpt[2][1],bbox[3])])
            points_norm = np.concatenate(([pnt0], [pnt1], [pnt2], [pnt3]))

            coordinate_dict = {
                "top-right": (points_norm[1][0]-x1, points_norm[1][1]-y1),
                "bottom-right": (points_norm[2][0]-x1, points_norm[2][1]-y1),
                "top-left": (points_norm[0][0]-x1, points_norm[0][1]-y1),
                "bottom-left": (points_norm[3][0]-x1, points_norm[3][1]-y1),
            }

            img_crop_lp = image_decode[y1:(y2), x1:x2]
            img_copy = img_crop_lp.copy()
            cropped_img = align_image(img_copy, coordinate_dict)

            img_list = []
            # detect and recognize in horizontal number plate
            check_plate_sqare_, img_list = check_plate_sqare(cropped_img)

            if check_plate_sqare_ is not None:
                img = check_plate_sqare_
            else:
                img = cropped_img
                img_list = [cropped_img]
            

            if self.MODE_REC:
                rec_res, _ = self.text_recognizer(img_list)
                txt_result, check_acc, arv_acc = mode_rec(rec_res, self.threshold)
                result_format_check = check_format_plate(txt_result)
                # result_format_check = True

                txt_result = check_format_plate_append(
                    txt_result
                )  # return lp + '#' or None

                if txt_result == None:
                    continue

            # if self.MODE_DET:
            #     dt_boxes, _ = self.text_detector(img)
            
            count_plate += 1
            results.append([id, txt_result, [x1, y1, x2, y2], [], [], []])

        return results, cropped_img
    

    def detect_license_plate_and_ocr(self, batch):
        results_det = self.detector.do_detect(batch)
        results_det_post = self.post_process_kpts(results_det)

        if not isinstance(batch, list):
            batch = [batch]
        
        final_result = []
        for result_det, raw_im in zip(results_det_post, batch):
            res, cropped_im = self.reg_plate(raw_im, result_det)
            final_result.append(res)
        
        return final_result

    def __call__(self, batch) -> Any:
        return self.detect_license_plate_and_ocr(batch)