import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

#lower image resolution while keeping image size constant i.e. val=0.5 = half the resolution
def lower_resolution(img, val):
    height,width,_=img.shape
    small=cv2.resize(img,(int(height*val),int(width*val)))
    return cv2.resize(small,(height,width))

def setrainstreak_in_image(img,xpos_start,ypos_start,xpos_stop,ypos_stop):
    cv2.line(img,(xpos_start,ypos_start),)


def simulate_rain_by_gaussian(img,vertical=2, vertical_variance=1,horizontal=4,horizontal_variance=1,lenstreaks=3,variance_lenstreaks=1,widthstreak=1, variance_widthstreak=0.5,angle=20,variance_angle=3):
    ypos_tot=0
    while ypos_tot<64:
        xpos_start=max(np.random.normal(horizontal,horizontal_variance),0)
        while xpos_start<64:
            ypos_start = max(np.random.normal(vertical, vertical_variance)+ypos_tot, 0)
            len=np.random.normal(lenstreaks,variance_lenstreaks)
            width=np.random.normal(widthstreak,variance_widthstreak)
            ang=np.random.normal(angle,variance_angle)
            xpos_stop=min(int(math.sin(ang*math.pi/180.0)),64)
            ypos_stop=min(int(math.cos(ang*math.pi/180.0)),64)
            setrainstreak_in_image(img, xpos_start,ypos_start,xpos_stop,ypos_stop,width)
            xpos_start+=np.random.normal(horizontal,horizontal_variance)
        ypos_tot+=np.random.normal(vertical, vertical_variance)


    temp=img
    for x in range(0,64):
        for y in range(0,64):
            for c in range(0,3):
                 temp[x][y][c]=int(noise[x][y][c]+img[x][y][c])
                # print type(temp[x][y][c])

    temp[temp > 255] = 255
    temp[temp < 0] = 0
    return temp

