import os

import numpy as np
import sys

BIRD_VIEW = "birdview"
FRONT_VIEW = "frontview"
SIDE_VIEW = "side_view"
MYCOBOT_PATH = "./assets/mycobot280.xml"
ROOT_DATA_DIR = r"D:\Osiris\Python\Projects\Sim2Real\MyCobotGym\mycobotgym\obj_localize\data"
MODEL_DIR = "trained_models"
# BEST_MODEL_PATH = "../vision_model/trained_models/domain_rand/VGGNet_epoch19.pth"
BEST_MODEL_PATH = "../vision_model/trained_models/reach_target/VGGNet_epoch49.pth"

DEFAULT_CAMERA_CONFIG = {
    "distance": 2,
    "azimuth": 0.0,
    "elevation": -35.0,
    "lookat": np.array([0, 0, 0.8]),
}

if __name__ == '__main__':
    print(sys.path)