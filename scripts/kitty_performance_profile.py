import sys
import os
import cProfile
import argparse
import kittty_visualization as kv

sys.path.append(os.path.abspath(os.path.join(".")))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Renders Kitti data with marked positions of objects')
    parser.add_argument('drive_number', type=int)
    args = parser.parse_args()
    def runVisualization():
        kv.runVisualization(args.drive_number)

    pr = cProfile.Profile()
    pr.enable()
    runVisualization()
    pr.disable()

    pr.print_stats(sort=1)