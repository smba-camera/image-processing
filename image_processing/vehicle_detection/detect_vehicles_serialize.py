import argparse
import cv2
import glob
import os
import sys
import pickle

# make modules accessible for the script
sys.path.append(os.path.abspath(os.path.join(".")))

import image_processing.vehicle_detection.Vehicle_detection as Vehicle_detection

def detect_vehicles_and_save(path, file_name):
    images = []
    for i in glob(os.path.join(path, '*.png')):
        images.append(cv2.imread(i))
    vehicle_detector = Vehicle_detection(images[0])
    vehicles = []
    for i in images:
        vehicles.append(vehicle_detector.find_vehicles(i))
    file_path = os.path.join('data', 'detected_vehicles', file_name)

    with open(file_path,'w') as f:
        pickle.dump(vehicles, f)

def load_detected_vehicles(file_name):
    file_path = os.path.join('data', 'detected_vehicles', file_name)
    with open(file_path, 'r') as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Renders Kitti data with marked positions of objects')
    parser.add_argument('images_path', type=int)
    parser.add_argument('file_name', type=int)
