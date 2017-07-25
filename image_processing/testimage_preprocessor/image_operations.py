import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

#lower image resolution while keeping image size constant i.e. val=0.5 = half the resolution
def lower_resolution(img, val):
    height,width,_=img.shape
    small=cv2.resize(img,(int(height*val),int(width*val)))
    return cv2.resize(small,(height,width))

def setrainstreak_in_image(img,xpos_start,ypos_start,xpos_stop,ypos_stop,width):
    print (xpos_start,ypos_start),(xpos_stop,ypos_stop),width
    cv2.line(img,(xpos_start,ypos_start),(xpos_stop,ypos_stop),color=(250,250,250),thickness=width)


def simulate_rain_by_gaussian(img,vertical=4, vertical_variance=1,horizontal=4,horizontal_variance=1,lenstreaks=3,variance_lenstreaks=1,widthstreak=1, variance_widthstreak=0.5,angle=20,variance_angle=3):
    ypos_tot=0
    while ypos_tot<64:
        xpos_start=max(int(np.random.normal(horizontal,horizontal_variance)),0)
        while xpos_start<64:
            ypos_start = max(int(np.random.normal(vertical, vertical_variance)+ypos_tot), 0)
            len=np.random.normal(lenstreaks,variance_lenstreaks)
            width=int(np.random.normal(widthstreak,variance_widthstreak))
            ang=np.random.normal(angle,variance_angle)
            xpos_stop=min(int(math.sin(ang*math.pi/180.0)),64)
            ypos_stop=min(int(math.cos(ang*math.pi/180.0)),64)
            setrainstreak_in_image(img, xpos_start,ypos_start,xpos_stop,ypos_stop,width)
            xpos_start+=int(np.random.normal(horizontal,horizontal_variance))
            plt.figure()
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.pause(5)
            plt.close()
        ypos_tot+=int(np.random.normal(vertical, vertical_variance))


    return img

