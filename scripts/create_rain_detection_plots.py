import os,sys
import glob
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import image_processing.vehicle_detection.Vehicle_detection as vd
from image_processing.testimage_preprocessor import image_operations
# make modules accessible for the script
sys.path.append(os.path.abspath(os.path.join(".")))

''' Creates plots for the detection rate of vehicles for images with rain '''

path = os.path.abspath(os.path.join('data', 'trainingSamples','*vehicles','*','*.png'))
detector=vd.VehicleDetection(np.zeros((64,64,3),np.uint8))
vallist= [20,18,16,14,12,10,8,6,4,2]
l=list(reversed(vallist))
classif_rate=[]
images=glob.glob(path)
for val in vallist:
    counter = 0
    rightClassifications = 0
    for image in images:
        if 'non' in image:
            is_car=0.0
        else:
            is_car=1.0
        img=cv2.imread(image)
        img=image_operations.simulate_rain_by_gaussian(img,vertical=val,horizontal=val)
        if detector.predict(img)[0] == is_car:
            rightClassifications += 1
        counter += 1
        if counter%1000==0:
            print counter
    print val,rightClassifications / float(counter)
    classif_rate.append(rightClassifications / float(counter))
for i in range(0,len(l)):
    l[i]*=5

plt.plot(l, classif_rate)
plt.axis([0, 100, 0.5, 1])
plt.xlabel('Strength of Rain')
plt.ylabel('correct classification of car vs. no-car')
plt.savefig('RainStrengthGraph.png')
plt.show()