import glob,os
import matplotlib.image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .vehicle_positions import VehiclePositions

class Visualizer:
    def __init__(self, camera_model, drive_num):
        self.camera_model = camera_model
        self.drive_num = drive_num

    def getVehicleColor(self, name):
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

    def showVisuals(self, path,date,CamNum='00'):
        fig=plt.figure()
        plt.get_current_fig_manager().window.state('zoomed')
        i=0
        vehiclePositions = VehiclePositions(path,date, self.drive_num)

        syncFolder = "{0}_drive_0002_sync".format(date)
        imgFolder = "image_{}".format(CamNum)
        imagePath = os.path.join(path, date, syncFolder, date, syncFolder, imgFolder,'data')
        img_glob = os.path.join(imagePath, "*.png")
        for pic in glob.glob(img_glob):
            if not plt.get_fignums():
                # window has been closed
                return
            img=matplotlib.image.imread(pic)
            #print ('new Image:')
            ax1=fig.add_subplot(211)
            ax1.imshow(img,cmap='gray')


            ax2=fig.add_subplot(212)

            vehicles=vehiclePositions.getVehiclePosition(i)
            count=len(vehicles)
            for j in range(count):
                name=vehicles[j][0]
                color=self.getVehicleColor(name)
                ax2.add_patch(patches.Rectangle((-vehicles[j][2]+vehicles[j][3], vehicles[j][1]-vehicles[j][4]),vehicles[j][3],vehicles[j][4],angle=vehicles[j][5],color=color) )
                vehicleCoord=[vehicles[j][1],vehicles[j][2],vehicles[j][6]]
                image_coords = self.camera_model.projectToImage(vehicleCoord)
                ax1.add_patch(patches.Rectangle(image_coords,20,20,color=color))
            #fig.draw
            ax2.set_ylim([0,100])
            ax2.set_xlim([-25,25])
            ax2.set_aspect(1)
            plt.pause(0.00000001)
            fig.clear()

            #time.sleep(10)
            i+=1

