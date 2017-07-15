import glob,os
import matplotlib.image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ..vehicle_positions import VehiclePositions

# Visualizes the real position of the cars

class GroundtruthVisualizer:
    def __init__(self, kitti, drive_num):
        self.camera_model = kitti.getVeloCameraModel()
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

        syncFolder = "{0}_drive_{1}_sync".format(date, self.drive_num)
        imgFolder = "image_{}".format(CamNum)
        imagePath = os.path.join(path, date, syncFolder, date, syncFolder, imgFolder,'data')
        img_glob = os.path.join(imagePath, "*.png")

        # load all images
        loaded_images = [matplotlib.image.imread(img) for img in glob.glob(img_glob)]

        for img in loaded_images:
            if not plt.get_fignums():
                # window has been closed
                return

            #print ('new Image:')
            ax1=fig.add_subplot(211)
            ax1.imshow(img,cmap='gray')


            ax2=fig.add_subplot(212)

            vehicles=vehiclePositions.getVehiclePosition(i)
            count=len(vehicles)
            for j in range(count):
                v = vehicles[j]
                name = v.type
                color=self.getVehicleColor(name)

                ax2.add_patch(patches.Rectangle((- v.yPos + v.width , v.xPos - v.length),v.width ,v.length ,angle=v.angle ,color=color) )
                vehicleCoord=[v.xPos ,v.yPos ,v.zPos]
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

