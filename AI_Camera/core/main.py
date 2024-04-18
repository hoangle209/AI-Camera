from omegaconf import OmegaConf, DictConfig
import os

from .detect.main import Detectors
from .track.main import Trackers

from AI_Camera.utils import get_pylogger
logger = get_pylogger()

class BaseModule:
    def __init__(self, config) -> None:
        self.cfg = self.load_config(config)
        self.configure_modules()

    def load_config(self, config):
        if isinstance(config, str) and config.endswith(("yaml", "yml")):
            if not os.path.exists(config):
                logger.warning(f"Path is not exist !!!. \
                               Using default config instead.")
                config = "configs/default.yaml"
            cfg = OmegaConf.load(config)
        elif isinstance(config, DictConfig):
            cfg = config
        else:
            logger.error(f"Expecting config is <{DictConfig.__name__}> instance or path to yaml file !!!")
            raise TypeError

        return cfg
    

    def configure_modules(self):
        self.supported_mode = []

        if "detection" in self.cfg:
            logger.info(f"Initiating <{self.cfg.detection.model}> detection module")
            self.detector = Detectors(self.cfg.detection)
            self.supported_mode.append("detection")
        
        if "track" in self.cfg:
            logger.info(f"Initiating <{self.cfg.track.model}> track module")
            self.tracker = Trackers(self.cfg.track)
            self.supported_mode.append("track")