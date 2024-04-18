import cv2
import numpy as np
from omegaconf import OmegaConf, DictConfig
import os

from AI_Camera.detect.main import Detectors
from AI_Camera.utils import get_pylogger
logger = get_pylogger()


class FireSmookeDetection:
    fps = 30
    def __init__(self, config) -> None:
        self.cfg = self.load_config(config)
        self.detector = Detectors(self.cfg.detection)

    def load_config(self, config):
        if isinstance(config, str) and config.endswith(("yaml", "yml")):
            if not os.path.exists(config):
                logger.warning(f"Path is not exist !!!. \
                               Using default config instead.")
                config = "configs/yolov8_bytetrack.yaml"
            cfg = OmegaConf.load(config)
        elif isinstance(config, DictConfig):
            cfg = config
        else:
            logger.error(f"Expecting config is <{DictConfig.__name__}> instance or path to yaml file !!!")
            raise TypeError

        return cfg

    def detect_fire_and_smooke(self, batch):
        dets = self.detector.do_detect(batch)
        return dets