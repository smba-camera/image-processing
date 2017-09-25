import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

#lower image resolution while keeping image size constant i.e. val=0.5 = half the resolution
def lower_resolution(img, val):
    height,width,_=img.shape
    small=cv2.resize(img,(int(height*val),int(width*val)))
    return cv2.resize(small,(height,width))

def setrainstreak_in_image(img,xpos_start,ypos_start,xpos_stop,ypos_stop,width,color):
    #print (xpos_start,ypos_start),(xpos_stop,ypos_stop),width
    cv2.line(img,(ypos_start,xpos_start),(ypos_stop,xpos_stop),color=color,thickness=width)


def simulate_rain_by_gaussian(img,vertical=10, vertical_variance=2,horizontal=10,horizontal_variance=2,lenstreaks=3,variance_lenstreaks=1,widthstreak=1, variance_widthstreak=0.5,angle=20,variance_angle=8,color=180,color_variance=2):
    width,height,channel=img.shape
    ypos_tot=0

    if(len(color)==1): color = (color,color,color)

    while ypos_tot<height:
        xpos_start=max(int(np.random.normal(3,horizontal_variance)),0)
        while xpos_start<width:
            ypos_start = max(int(np.random.normal(3, vertical_variance)+ypos_tot), 0)
            streak_len=np.random.normal(lenstreaks,variance_lenstreaks)
            linewidth=max(int(np.random.normal(widthstreak,variance_widthstreak)),0)
            ang=np.random.normal(angle,variance_angle)
            xpos_stop=xpos_start+min(int(math.cos(ang*math.pi/180.0)*streak_len),width)
            ypos_stop=ypos_start+min(int(math.sin(ang*math.pi/180.0)*streak_len),height)
            R = min(max(int(np.random.normal(color[0],color_variance)),0),255)
            G = min(max(int(np.random.normal(color[1], color_variance)), 0), 255)
            B = min(max(int(np.random.normal(color[2], color_variance)), 0), 255)
            setrainstreak_in_image(img, xpos_start,ypos_start,xpos_stop,ypos_stop,linewidth,(R,G,B))
            xpos_start+=int(np.random.normal(horizontal,horizontal_variance))

        ypos_tot+=int(np.random.normal(vertical, vertical_variance))


    return img

