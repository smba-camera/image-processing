import os
import glob,sys
import cv2

# make modules accessible for the script
sys.path.append(os.path.abspath(os.path.join(".")))

import image_processing.vehicle_detection.Vehicle_detection as vd

def runVisualization():
        sampleimg=cv2.imread('data/images2/0000000000.png')
        detector=vd.VehicleDetection(sampleimg)
        images=glob.glob('data/images2/*.png')
        images.sort()
        for image in images:
            img=cv2.imread(image)
            detector.find_vehicles(img)

if __name__ == "__main__":
    runVisualization()
