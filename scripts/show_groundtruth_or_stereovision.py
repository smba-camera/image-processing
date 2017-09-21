import os
import sys
# make modules accessible for the script
sys.path.append(os.path.abspath(os.path.join(".")))

from image_processing.kitti_data import Kitti
from image_processing.kitti_data import GroundtruthVisualizer, RangeestimationVisualizer
import argparse

''' shows the data from the kitti dataset along with range estimations for the vehicles '''

def runVisualization(drive_num, visualizationType):
    path = os.path.abspath(os.path.join('data', 'kitti'))
    Dates = ['2011_09_26']

    drive_num = "{0:04d}".format(drive_num)

    for date in Dates:
        kitti = Kitti(path, date)

        Visualizer = createVisualizer(visualizationType, kitti, drive_num)
        Visualizer.showVisuals(path, date)

def createVisualizer(visualizationType, kitti, drive_num):
    if visualizationType == 1:
        return RangeestimationVisualizer(kitti, drive_num)
    return GroundtruthVisualizer(kitti, drive_num)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Renders Kitti data with marked positions of objects')
    parser.add_argument('drive_number', type=int)
    parser.add_argument('visulizationType', type=int,help='0: groundTruth, 1: rangeestimation')
    args = parser.parse_args()
    runVisualization(args.drive_number, args.visulizationType)
