import os
import numpy as np
import cv2

fgbg = cv2.createBackgroundSubtractorMOG2()
kernel1 = np.ones((2,2),np.uint8)
kernel2 = np.ones((2,2),np.uint8)

images_dir = "C:\Users\Paul\Desktop\SS2017\SMBAD\source\scripts\weather-image\data"

images = []
for file in os.listdir(images_dir):
    if file.endswith("jpg"):
        images.append(images_dir+"\\"+file)

for image in images:
    frame = cv2.imread(image)
    fgmask = fgbg.apply(frame)
    img = fgmask
    img = cv2.erode(img,kernel1,iterations = 1)
    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel2)
    cv2.imshow("video", img)
    #uncomment this to see frame by frame
    cv2.waitKey(0)    

cv2.waitKey(0)
cv2.destroyAllWindows()