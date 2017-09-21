import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(".")))
from image_processing.vehicle_detection.detect_vehicles_serialize import detect_vehicles_and_save

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Renders Kitti data with marked positions of objects')
    parser.add_argument('images_path')
    parser.add_argument('file_name')
    parser.add_argument('-min_frame', type=int, default=0)
    parser.add_argument('-max_frame', type=int, default=0)

    #parser.add_argument('path')
    args = parser.parse_args()
    detect_vehicles_and_save(args.images_path, args.file_name,args.min_frame, args.max_frame)
    #print(load_detected_vehicles(args.path))
