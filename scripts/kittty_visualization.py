import os
import sys
# make modules accessible for the script
sys.path.append(os.path.abspath(os.path.join(".")))

from image_processing.kitti_data import Kitti
from image_processing.kitti_data import visualize
import argparse

def runVisualization(drive_num):
    path = os.path.abspath(os.path.join('data', 'kitti'))
    Dates = ['2011_09_26']

    drive_num = "{0:04d}".format(drive_num)

    for date in Dates:
        data = Kitti.Kitti()
        model = data.initialize(path, date)
        Visualizer = visualize.Visualizer(model, drive_num)
        Visualizer.showVisuals(path, date)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Renders Kitti data with marked positions of objects')
    parser.add_argument('drive_number', type=int)
    args = parser.parse_args()
    runVisualization(args.drive_number)
