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
        for data in self.label_dict:
            if len(data) == 3:
                # meter -> centimeter
                label = np.array([x * 100 for x in data[2]], dtype=np.float32)
                self.dataset.append((data[0], data[1], label))
            else:
                label = np.array([x * 100 for x in data[1]], dtype=np.float32)
                self.dataset.append((data[0], label))

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if len(data) == 3:
            img1_name = data[0]
            img2_name = data[1]
            label = data[2]
            img1_path = convert_path(os.path.join(self.file_dir, img1_name))
            img2_path = convert_path(os.path.join(self.file_dir, img2_name))
            image1 = Image.open(img1_path)
            image1 = image1.resize((224, 224))
            image2 = Image.open(img2_path)
            image2 = image2.resize((224, 224))
            if self.transform:
                image1 = self.transform(image1)
                image2 = self.transform(image2)
            return image1, image2, label
        else:
            img_name = self.dataset[idx][0]
            label = self.dataset[idx][1]
            img_path = convert_path(os.path.join(self.file_dir, img_name))
            image = Image.open(img_path)
            image = image.resize((224, 224))
            if self.transform:
                image = self.transform(image)
            return image, label

    def __len__(self):
        return len(self.dataset)



