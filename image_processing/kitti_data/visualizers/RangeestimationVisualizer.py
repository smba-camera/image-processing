import glob,os
import matplotlib.image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ..vehicle_positions import VehiclePositions
import cv2


# detected cars within the kitti images and estimations for distances

class RangeestimationVisualizer:
    def __init__(self, kitti, drive_num):
        self.camera_model_velo = kitti.getVeloCameraModel()
        self.camera_model_1 = kitti.getCameraModel(0)
        self.camera_model_2 = kitti.getCameraModel(1)
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

    def getRealVehiclePositions(self, imgId, vehiclePositions):
        ''' finds vehicles within the image. Will just '''
        return vehiclePositions.getVehiclePosition(imgId)

    def findVehiclesOnImages(self, img1, img2, realVehiclePositions=None):
        '''Returns a list of tuples (RectancleOnImg1, RectanleOnImg2)'''
        img_coords = []

        # fake implementation for the time the implementation of the detection of vehicles is not finished
        for v in realVehiclePositions:
            img_coord_1 = self.camera_model_velo.projectToImage([v.xPos, v.yPos, v.zPos])
            img_coord_2 = self.camera_model_velo.projectToImage([v.xPos, v.yPos, v.zPos])
            img_coords.append((img_coord_1, img_coord_2))

        return img_coords

    def showVisuals(self, path,date,CamNum='00'):
        fig=plt.figure()
        plt.get_current_fig_manager().window.state('zoomed')
        i=0
        vehiclePositions = VehiclePositions(path,date, self.drive_num)

        syncFolder = "{0}_drive_{1}_sync".format(date, self.drive_num)
        imgFolder = os.path.join(path, date, syncFolder, date, syncFolder, "image_{}",'data')
        cam0_imgPath = imgFolder.format('00')
        cam1_imgPath = imgFolder.format('01')
        images_camera_0 = map(lambda img: os.path.join(cam0_imgPath, img), os.listdir(cam0_imgPath))
        images_camera_1 = map(lambda img: os.path.join(cam1_imgPath, img), os.listdir(cam1_imgPath))

        # check that for every frame of one camera there is an image from the other
        assert(len(images_camera_0) == len(images_camera_1))

        # load all images:
        print("Loading images...\n")
        loaded_images = []
        for (pic0, pic1) in zip(images_camera_0, images_camera_1):
            #RGB BGR
            loaded_pic_0 = cv2.imread(pic0, cv2.IMREAD_UNCHANGED)
            #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            loaded_images.append((loaded_pic_0, None))

        print("Done.\nShow Range estimations...\n")

        for (img0, img1) in loaded_images:
            if not plt.get_fignums():
                # window has been closed
                return

            #print ('new Image:')
            ax1=fig.add_subplot(211)
            ax1.imshow(img0,cmap='gray')


            ax2=fig.add_subplot(212)

            vehicles = vehiclePositions.getVehiclePosition(i)
            count = len(vehicles)
            for j in range(count):
                v = vehicles[j]
                name = v.type
                color = self.getVehicleColor(name)

                ax2.add_patch(
                    patches.Rectangle((- v.yPos + v.width, v.xPos - v.length), v.width, v.length, angle=v.angle,
                                      color=color))
                vehicleCoord = [v.xPos, v.yPos, v.zPos]
                image_coords = self.camera_model_velo.projectToImage(vehicleCoord)
                ax1.add_patch(patches.Rectangle(image_coords, 20, 20, color=color))

            #fig.draw
            ax2.set_ylim([0,100])
            ax2.set_xlim([-25,25])
            ax2.set_aspect(1)
            plt.pause(0.00000001)
            fig.clear()

            #time.sleep(10)
            i+=1

