import numpy as np
import time
import cv2
import glob,os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ExtractXMLData
import calibration

def getVehicleColor(name):
    color='none'
    if name == 'Car':
        color = 'red'
    elif name == 'Van':
        color = 'orange'
    elif name == 'Truck':
        color = 'yellow'
    elif name == 'Tram':
        color = 'blue'
    elif name == 'Cyclist':
        color = 'green'
    elif name == 'Pedestrian':
        color = 'black'
    elif name == 'Person':
        color = 'brown'
    elif name == 'Misc':
        color = 'grey'
    return color

def initVisualize(path,date,CamNum='00'):
    os.chdir(path+'\\'+date+'\\'+date+'_drive_0001_sync\\'+date+'\\'+date+'_drive_0001_sync\\image_'+CamNum+'\data')

def showVisuals(path,date):
    initVisualize(path,date)
    fig=plt.figure()
    plt.get_current_fig_manager().window.state('zoomed')
    i=0
    ExtractXMLData.initXMLReader(path,date)
    for pic in glob.glob("*.png"):
        img=cv2.imread(pic,0)
        #print ('new Image:')
        ax1=fig.add_subplot(211)
        ax1.imshow(img,cmap='gray')


        ax2=fig.add_subplot(212)

        vehicles=ExtractXMLData.getVehiclePosition(i)
        count=len(vehicles)
        for j in range(count):
            name=vehicles[j][0]
            color=getVehicleColor(name)
            ax2.add_patch(patches.Rectangle((-vehicles[j][2]+vehicles[j][3], vehicles[j][1]-vehicles[j][4]),vehicles[j][3],vehicles[j][4],angle=vehicles[j][5],color=color) )
            vehicleCoord=[vehicles[j][1],vehicles[j][2],vehicles[j][6],1]
            print vehicleCoord
            ax1.add_patch(patches.Rectangle((calibration.getImageCoordinates(vehicleCoord)),20,20,color=color))
        #fig.draw
        ax2.set_ylim([0,100])
        ax2.set_xlim([-25,25])
        ax2.set_aspect(1)
        plt.pause(0.001)
        ax1.remove()
        plt.draw()
        plt.clf()

        #time.sleep(10)
        i+=1
        print i
