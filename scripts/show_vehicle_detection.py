import os
import glob,sys
import cv2
import argparse

# make modules accessible for the script
sys.path.append(os.path.abspath(os.path.join(".")))

import image_processing.vehicle_detection.Vehicle_detection as vd

''' uses vehicle detection to mark all vehicles on the kitti images '''

def runVisualization(path):
    path = path
    sampleimg=cv2.imread(glob.glob(os.path.join(path,'*.jpg'))[0])
    detector=vd.VehicleDetection(sampleimg)
    images=glob.glob(os.path.join(path,'*.png'))
    images.sort()
    for image in images:
        img=cv2.imread(image)
        detector.show_vehicles(img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shows detected vehicles')
    parser.add_argument('path')
    args = parser.parse_args()
    runVisualization(args.path)
