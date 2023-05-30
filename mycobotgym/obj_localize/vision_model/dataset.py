import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from PIL import Image
from mycobotgym.obj_localize.utils.general_utils import *


class TrainDataset(Dataset):
    def __init__(self, file_dir, transform=None):
        super(TrainDataset, self).__init__()
        self.dataset = []
        self.file_dir = os.path.join(file_dir, "data")
        self.transform = transform
        dict_path = convert_path(os.path.join(file_dir, "pos_map.json"))
        with open(dict_path, 'r') as f:
            self.label_dict = json.load(f)
        f.close()
        for image in os.listdir(self.file_dir):
            label = self.label_dict.get(image)
            # label = np.array(label, dtype=np.float32)
            # meter -> centimeter
            label = np.array([x*100 for x in label], dtype=np.float32)
            self.dataset.append((image, label))

    def __getitem__(self, idx):
        img_name = self.dataset[idx][0]
        label = self.dataset[idx][1]
        # only estimate x-y position
        # label = label[:-1]
        img_path = convert_path(os.path.join(self.file_dir, img_name))
        # image = plt.imread(img_path)
        image = Image.open(img_path)
        image = image.resize((224, 224))
        # display_img_plt(image)
        # image = resize_img(image)
        # display_img_cv2(image)
        if self.transform:
            image = self.transform(image)
        # print(image)
        return image, label

    def __len__(self):
        return len(self.dataset)



