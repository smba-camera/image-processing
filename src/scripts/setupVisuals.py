import visualize
import calibration

path='C:\Users\Daniel\Documents\School\TU\mbad\Data'
Dates=['2011_09_26']

for date in Dates:
    calibration.initialize(path,date)
    visualize.showVisuals(path,date)


