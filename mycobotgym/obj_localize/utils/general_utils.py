import numpy as np
import torch
import os
import wandb
import matplotlib.pyplot as plt

from torchvision.models import vgg16


def convert_path(path):
    return path.replace('\\', '/')


def wandb_init():
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="MyCobot_Summer2023",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 1e-4,
            "architecture": "VGG16",
            "dataset": "domain-rand",
            "epochs": 10,
        },
        resume=False
    )
    return run


def count_files(file_dir):
    for _, _, filenames in os.walk(file_dir):
        return len(filenames)


def check_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def load_weights(path):
    if os.path.exists(path):
        return torch.load(path)
    vgg16_weights = vgg16(pretrained=True).state_dict()
    torch.save(vgg16_weights, path)
    return vgg16_weights


def plot(pth1, pth2):
    train_err = np.loadtxt(pth1)
    val_err = np.loadtxt(pth2)
    # assert len(train_err) == len(val_err)
    n = len(train_err)
    plt.title("Training & Validation Error")
    plt.plot(range(1, n+1), train_err, label="Train")
    plt.plot(range(1, n+1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    # pth1 = "D:\Osiris\Python\Projects\Sim2Real\data\deprecate\\25b-30e-freeze\VGGNet_train_err.csv"
    # pth2 = "D:\Osiris\Python\Projects\Sim2Real\data\deprecate\\25b-30e-freeze\VGGNet_val_err.csv"
    pth1 = "D:\Osiris\Python\Projects\Sim2Real\MyCobotGym\mycobotgym\obj_localize\\vision_model\\trained_models" \
           "\\reach_target\VGGNet_train_err.csv"
    pth2 = "D:\Osiris\Python\Projects\Sim2Real\MyCobotGym\mycobotgym\obj_localize\\vision_model\\trained_models" \
           "\\reach_target\VGGNet_val_err.csv"
    plot(pth1, pth2)