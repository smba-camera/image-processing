import os
import sys
# make modules accessible for the script
sys.path.append(os.path.abspath(os.path.join(".")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from image_processing.kitti_data import Kitti, GroundtruthVisualizer, RangeestimationVisualizer
import argparse

def save_video(vis_type, drive_num, video_name, start_frame=0, end_frame=0):
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
    kitti_drive_num = "{:04d}".format(drive_num)
    kitti = Kitti(kitti_path, kitti_date)
    if (vis_type == 0):
        visualizer = GroundtruthVisualizer(kitti, kitti_drive_num, yield_frames=True)
    if (vis_type == 1):
        visualizer = RangeestimationVisualizer(kitti, kitti_drive_num,
            start_frame=start_frame, end_frame=end_frame)

    visual_generator = visualizer.showVisuals_generator(kitti_path, kitti_date) # is yielded
    fig = visual_generator.next()

    file_name = os.path.join(path, "{}.mp4".format(video_name))
    with writer.saving(fig, file_name, 100):
        for _ in visual_generator:
            writer.grab_frame()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Renders Kitti data with marked positions of objects')

    parser.add_argument('drive_number', type=int)
    parser.add_argument('visualizationType', type=int,help='0: groundTruth, 1: rangeestimation')
    parser.add_argument('video_name')
    parser.add_argument('-sf', '--start_frame', type=int)
    parser.add_argument('-ef', '--end_frame', type=int)

    args = parser.parse_args()
    save_video(
        args.visualizationType,
        args.drive_number,
        args.video_name,
        start_frame=args.start_frame,
        end_frame=args.end_frame)
