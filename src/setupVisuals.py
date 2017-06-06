from image_processing.kitti_data import Kitti
from image_processing.kitti_data import visualize

path='C:\Users\Daniel\Documents\School\TU\mbad\Data'
Dates=['2011_09_26']

for date in Dates:
    data=Kitti.Kitti()
    model=data.initialize(path,date)
    Visualizer=visualize.Visualizer(model)
    Visualizer.showVisuals(path,date)

