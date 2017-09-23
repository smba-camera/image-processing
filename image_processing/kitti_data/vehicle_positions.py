import xml.etree.ElementTree as ET
from collections import namedtuple
from Kitti import Kitti
import os

class Vehicle:
    def __init__(self, type, xPos, yPos, zPos, width, length, angle):
        self.type = type
        self.xPos = xPos
        self.yPos = yPos
        self.zPos = zPos
        self.width = width
        self.length = length
        self.angle = angle

class VehiclePositions:
    def __init__(self, path, date, drive_num):
        trackletsFolder = "{0}_drive_{1}_tracklets".format(date, drive_num)
        syncFolder = "{0}_drive_{1}_sync".format(date, drive_num)
        trackletsPath = os.path.join(path, date, trackletsFolder, date, syncFolder)
        filePath = os.path.join(trackletsPath, 'tracklet_labels.xml')
        self.parsed_xml = ET.parse(filePath).getroot()
        self.kitti = Kitti(path,date)

    def get_frame_count(self):
        e = self.parsed_xml
        count = int(e[0][0].text)
        return count

    def getVehiclePosition(self, frame):
        e = self.parsed_xml

        count = self.get_frame_count()
        vehicles=[]
        for index in range(2,count+2):
            startFrame=int (e[0][index][4].text)
            endFrame=startFrame + int (e[0][index][5][0].text) -1
            if frame>=startFrame and frame<=endFrame:
                name = e[0][index][0].text
                xPos=float (e[0][index][5][2+frame-startFrame][0].text)
                yPos = float (e[0][index][5][2+frame-startFrame][1].text)
                zPos= float (e[0][index][5][2+frame-startFrame][2].text)
                width = float (e[0][index][2].text)
                length =float (e[0][index][3].text)
                angle= float (e[0][index][5][2+frame-startFrame][5].text)
                newVehicle = Vehicle(type=name, xPos=xPos, yPos=yPos, width=width, length=length, angle=angle, zPos=zPos)
                vehicles.append(newVehicle)
        return vehicles

    '''Projected to the coordinate system of camera 0'''
    def getVehiclePositions_projected(self, frame):
        vehicles = self.getVehiclePosition(frame)
        velo_extrinsic_model = self.kitti.getVeloExtrinsicModel()
        cars = []
        for v in range(len(vehicles)):
            if vehicles[v].type in ('Car'):  # ,'Van','Truck'):
                x, y, z = velo_extrinsic_model.project_coordinates([vehicles[v].xPos, vehicles[v].yPos, vehicles[v].zPos])
                cars.append((x,y,z))
        return cars

