import os
import sys
# make modules accessible for the script
sys.path.append(os.path.abspath(os.path.join(".")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from image_processing.kitti_data import Kitti, GroundtruthVisualizer
import argparse

def save_video():#vis_type, drive_num
    # todo move into library so it can be used by other scripts

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=7, metadata=metadata)
    path = os.path.abspath(os.path.join('data', 'videos'))
    if not os.path.exists(path):
        os.mkdir(path)

    # kitti stuff
    kitti_path = os.path.join('data','kitti')
    kitti_date = '2011_09_26'
    kitti_drive_num = '0001'
    kitti = Kitti(kitti_path, kitti_date)
    visualizer = GroundtruthVisualizer(kitti, kitti_drive_num, yield_frames=True)

    visual_generator = visualizer.showVisuals_generator(kitti_path, kitti_date) # is yielded
    fig = visual_generator.next()

    file_name = os.path.join(path, 'writer_test.mp4')
    with writer.saving(fig, file_name, 100):
        for _ in visual_generator:
            writer.grab_frame()
    writer.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Renders Kitti data with marked positions of objects')
    #parser.add_argument('drive_number', type=int)
    #parser.add_argument('visulizationType', type=int,help='0: groundTruth, 1: rangeestimation')
    args = parser.parse_args()
    save_video()
