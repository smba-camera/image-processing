import xml.etree.ElementTree as ET
from collections import namedtuple

class VehiclePositions:
    def __init__(self, path, date):
        self.parsed_xml = ET.parse(path + '\\' + date + '\\' + date + '_drive_0001_tracklets\\' + date + '\\' + date + '_drive_0001_sync\\tracklet_labels.xml').getroot()

    def getVehiclePosition(self, frame):
        e = self.parsed_xml

        count=int (e[0][0].text)
        vehicle = namedtuple('vehicle','type xPos yPos width length angle zPos')
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
                vehicles.append(vehicle(type=name,xPos=xPos,yPos=yPos,width=width,length=length,angle=angle,zPos=zPos))
        return vehicles


