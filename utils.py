import torch
import cv2


def to_gray_scale_and_downsample(img):
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img