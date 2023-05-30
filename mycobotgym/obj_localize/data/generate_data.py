import json
import os
import sys
import argparse
from tqdm import tqdm

from sim_manager import SimManager
from PIL import Image
from mycobotgym.obj_localize.utils.general_utils import *
from mycobotgym.obj_localize.constants import *


def _initialize(path):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump({}, f)


def generate_data(file_name, num_img):
    sim_manager = SimManager(MYCOBOT_PATH)
    data_dir = convert_path(os.path.join(ROOT_DATA_DIR, file_name))
    check_exists(data_dir)
    img_dir = convert_path(os.path.join(data_dir, "data"))
    check_exists(img_dir)
    pos_path = convert_path(os.path.join(data_dir, "pos_map.json"))
    _initialize(pos_path)
    img_count = count_files(img_dir)
    print("Number of images in dataset: {}".format(img_count))
    for i in tqdm(range(num_img)):
        pos, img = sim_manager.get_data(BIRD_VIEW)
        img_name = str(i) + ".jpg"
        write_data(pos, img, img_dir, pos_path, img_name)


def write_data(pos, img, img_dir, pos_path, image_name):
    save_img = Image.fromarray(img)
    img_path = convert_path(os.path.join(img_dir, image_name))
    save_img.save(img_path)
    pos_data = {image_name: pos.tolist()}
    with open(pos_path, 'r') as f:
        json_data = json.load(f)
        json_data.update(pos_data)
        # print(json_data)
    with open(pos_path, 'w') as f_new:
        json.dump(json_data, f_new)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_images', type=int, default=1)
    args = parser.parse_args()
    generate_data("domain_rand", args.num_images)
