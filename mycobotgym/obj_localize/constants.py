import numpy as np

BIRD_VIEW = "birdview"
MYCOBOT_PATH = "./assets/mycobot280.xml"
ROOT_DATA_DIR = "data"
MODEL_DIR = "vision_model/trained_models"
BEST_MODEL_PATH = "../vision_model/trained_models/domain_rand/VGGNet_epoch19.pth"

DEFAULT_CAMERA_CONFIG = {
    "distance": 2,
    "azimuth": 0.0,
    "elevation": -35.0,
    "lookat": np.array([0, 0, 0.8]),
}