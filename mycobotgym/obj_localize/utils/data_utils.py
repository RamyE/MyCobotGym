import cv2
import numpy as np
import matplotlib.pyplot as plt


def sample_xyz(range):
    x = np.random.uniform(range[0][0], range[0][1])
    y = np.random.uniform(range[1][0], range[1][1])
    z = np.random.uniform(range[2][0], range[2][1])
    return (x, y, z)


def sample_quat(range):
    yaw = np.random.uniform(range[0][0], range[0][1]) * np.pi / 180
    pitch = np.random.uniform(range[1][0], range[1][1]) * np.pi / 180
    roll = np.random.uniform(range[2][0], range[2][1]) * np.pi / 180
    return [yaw, pitch, roll]


def sample_color(range):
    r = np.random.uniform(range[0], range[1]) / 255
    g = np.random.uniform(range[0], range[1]) / 255
    b = np.random.uniform(range[0], range[1]) / 255
    return [r, g, b]


def sample_fov(range):
    return np.random.uniform(range[0], range[1])


def display_img(image):
    plt.imshow(image)
    plt.show()

