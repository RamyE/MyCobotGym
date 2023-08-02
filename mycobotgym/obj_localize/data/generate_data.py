import json
import os
import sys
import argparse
from tqdm import tqdm

from PIL import Image
from mycobotgym.obj_localize.envs.mycobot_vision import MyCobotVision
from mycobotgym.obj_localize.utils.general_utils import *
from mycobotgym.obj_localize.constants import *


def initialize(path):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump([], f)


def generate_data(file_name, num_img, cam1=BIRD_VIEW, cam2=None):
    env = MyCobotVision()
    check_exists(file_name)
    img_dir = convert_path(os.path.join(file_name, "data"))
    check_exists(img_dir)
    pos_path = convert_path(os.path.join(file_name, "pos_map.json"))
    initialize(pos_path)
    img_count = count_files(img_dir)
    if cam2 is not None:
        img_count = int(img_count / 2)
    print("Number of images in dataset: {}".format(img_count))
    for i in tqdm(range(num_img)):
        if cam2 is not None:
            idx = i + img_count
            pos, img1, img2 = env.generate_image(cam1, cam2)
            write_data(pos, img1, img_dir, pos_path, idx, img2)
        else:
            idx = i + img_count
            pos, img = env.generate_image(cam1)
            write_data(pos, img, img_dir, pos_path, idx)


def write_data(pos, img1, img_dir, pos_path, idx, img2=None):
    if img2 is not None:
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        img1_name = f"{idx}_a.jpg"
        img2_name = f"{idx}_b.jpg"
        img1_path = convert_path(os.path.join(img_dir, img1_name))
        img2_path = convert_path(os.path.join(img_dir, img2_name))
        img1.save(img1_path)
        img2.save(img2_path)
        # pos_data = {[img1_name, img2_name]: pos.tolist()}
        pos_data = [img1_name, img2_name, pos.tolist()]
    else:
        img = Image.fromarray(img1)
        img_name = f"{idx}.jpg"
        img_path = convert_path(os.path.join(img_dir, img_name))
        img.save(img_path)
        # pos_data = {[img_name]: pos.tolist()}
        pos_data = [img_name, pos.tolist()]

    with open(pos_path, 'r') as f:
        json_data = json.load(f)
        # json_data.update(pos_data)
        json_data.append(pos_data)
        update_data = json.dumps(json_data)
        # print(json_data)
    with open(pos_path, 'w') as f_new:
        # json.dump(json_data, f_new)
        f_new.write(update_data)


if __name__ == '__main__':
    generate_data("test_bird_front", 500, BIRD_VIEW, FRONT_VIEW)
