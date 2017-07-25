import os,sys
import glob
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import image_processing.vehicle_detection.Vehicle_detection as vd
from image_processing.testimage_preprocessor import image_operations
# make modules accessible for the script
sys.path.append(os.path.abspath(os.path.join(".")))


path = os.path.abspath(os.path.join('..','data', 'trainingSamples','*vehicles','*','*.png'))
detector=vd.VehicleDetection(np.zeros((64,64,3),np.uint8))
vallist= [1,1/2.0,1/4.0,1/8.0,1/16.0,1/32.0,1/64.0]
classif_rate=[]
for val in vallist:
    counter = 0
    rightClassifications = 0
    for image in glob.glob(path):
        if 'non' in image:
            is_car=0.0
        else:
            is_car=1.0
        img=cv2.imread(image)
        img=image_operations.lower_resolution(img,val)

        #print detector.predict(img)[0],is_car
        if detector.predict(img)[0] == is_car:
            rightClassifications+=1
        counter+=1
    classif_rate.append(rightClassifications/float(counter))
for i in range(0,len(vallist)):
    vallist[i]*=64
plt.plot(vallist,classif_rate)
plt.axis([1,64,0.5,1])
plt.savefig('resolutionGraph.png')
plt.show()
