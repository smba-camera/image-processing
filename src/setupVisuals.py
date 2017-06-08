import os
from image_processing.kitti_data import Kitti
from image_processing.kitti_data import visualize
import argparse

parser = argparse.ArgumentParser(description='Renders Kitti data with marked positions of objects')
parser.add_argument('drive_number', type=int)
args = parser.parse_args()

path=os.path.abspath(os.path.join('..', 'data','kitti'))
Dates=['2011_09_26']

drive_num = "{0:04d}".format(args.drive_number)

for date in Dates:
    data=Kitti.Kitti()
    model=data.initialize(path,date)
    Visualizer=visualize.Visualizer(model, drive_num)
    Visualizer.showVisuals(path,date)

