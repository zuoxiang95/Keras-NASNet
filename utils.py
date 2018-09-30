# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from keras.utils import np_utils
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data(data_line, input_shape, num_classes, random=True, jitter=0.3, sat=1.5, val=1.5, proc_img=True):
    """
        random processing for real-time data augmentation
    :param data_line:
    :param input_shape:
    :param random:
    :param jitter:
    :param sat:
    :param val:
    :param proc_img:
    :return:
    """
    line = data_line.split(',')
    image = Image.open(line[0])
    h, w = input_shape
    label = np_utils.to_categorical(line[1], num_classes)

    if not random:
        image = image.resize((h, w), Image.BICUBIC)
        image_data = np.array(image) / 255.
        return image_data, label

    image = image.resize((h, w), Image.BICUBIC)

    # flip image
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)

    return image_data, label
