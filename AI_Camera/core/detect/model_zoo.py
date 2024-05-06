"""
    This module contains detect function of detectors
    Return List[(xyxy, conf, cls)]
"""

def do_detect_yolov8(
        v8_model=None, 
        batch=None,
        imgsz=(640, 640),
        conf=0.25,
        classes=(0,),
        verbose=False,
        device="cpu",
        agnostic_nms=True
    ):

    v8_preds = v8_model(
                    batch, 
                    imgsz=imgsz, 
                    conf=conf, 
                    classes=classes, 
                    verbose=verbose, 
                    device=device,
                    agnostic_nms=agnostic_nms 
                )   
    
    results = [pred.boxes.data.cpu().numpy() for pred in v8_preds]
    if v8_preds[0].keypoints is not None:
        keypoints = [pred.keypoints.data.cpu().numpy() for pred in v8_preds]
        results = (results, keypoints)

    return results


def do_detect_yolov5(
        v5_model=None, 
        batch=None,
        imgsz=(640, 640),
        conf=0.25,
        classes=(0,),
        verbose=False,
        device="cpu",
        agnostic_nms=True
    ):

    v5_model.conf = conf
    v5_model.classes = classes
    v5_model.verbose = verbose
    v5_model.agnostic_nms = agnostic_nms
    v5_model = v5_model.to(device)

    batch = [im[..., ::-1] for im in batch] # yolov5 expects RGB input
    v5_preds = v5_model(batch, size=imgsz)
    results = [pred.cpu().numpy() for pred in v5_preds.xyxy]
    return results


detector_zoo = {
    "yolov8": do_detect_yolov8,
    "yolov5": do_detect_yolov5
}