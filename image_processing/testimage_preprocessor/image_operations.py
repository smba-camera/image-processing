import numpy as np
import cv2
import matplotlib.pyplot as plt

#lower image resolution while keeping image size constant i.e. val=0.5 = half the resolution
def lower_resolution(img, val):
    height,width,_=img.shape
    small=cv2.resize(img,(int(height*val),int(width*val)))
    return cv2.resize(small,(height,width))

