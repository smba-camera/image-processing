import argparse
import cv2
import glob
import os
import sys
import pickle
from datetime import datetime
from Vehicle_detection import VehicleDetection
#import image_processing.vehicle_detection.VehicleDetection as VehicleDetection

def detect_vehicles_and_save(path, file_name, min_frame, max_frame):
    start_time = datetime.now()
    images = []

    images = sorted(glob.glob(os.path.join(path, '*.png')))

    vehicle_detector = VehicleDetection(cv2.imread(images[0]))
    vehicles = []

    counter = 1
    images = images[min_frame:max_frame] if max_frame else images[min_frame:]
    print("amount of images: {}, takes about {}m".format(len(images), len(images)/0.75))
    for i_path in images:
        img = cv2.imread(i_path)
        vehicles.append(vehicle_detector.find_vehicles(img))
        now = datetime.now()
        remaining = (now-start_time)/counter*(len(images)-counter)
        ready_time = now+remaining
        print("{}/{} time remaining: {} ready at: {}".format(counter,len(images), remaining, ready_time))
        counter += 1

    path = os.path.join('data', 'detected_vehicles')
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path, file_name)

    with open(file_path,'wb') as f:
        pickle.dump(vehicles, f)

def load_detected_vehicles(file_name):
    file_path = os.path.join('data', 'detected_vehicles', file_name)
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def get_detected_vehicles_file_name(drive, img_num, threshold):
    return "{:04d}_{:02d}_t{}".format(drive,img_num, threshold)
