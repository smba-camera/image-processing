import os
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ..vehicle_positions import VehiclePositions
import cv2
import sys
import image_processing.util.Util as util


# detected cars within the kitti images and estimations for distances

class GroundtruthVisualizer:
    def __init__(self, kitti, drive_num, yield_frames=False):
        self.camera_model_velo_camera_0 = kitti.getVeloCameraModel()
        self.camera_model_1 = kitti.getCameraModel(0)
        self.camera_model_2 = kitti.getCameraModel(1)
        self.drive_num = drive_num
        self.yield_frames = yield_frames

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
            img_coord_0 = self.camera_model_velo_camera_0.projectToImage([v.xPos, v.yPos, v.zPos])
            # TODO: next line is currently wrong (projection to image0)
            img_coord_1 = self.camera_model_velo_camera_0.projectToImage([v.xPos, v.yPos, v.zPos])
            img_coords.append((img_coord_0, img_coord_1))

        return img_coords

    def showVisuals(self, path,date):
        fig=plt.figure()
        #plt.get_current_fig_manager().window.state('zoomed')
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
        sys.stdout.write("Loading images...")
        start_time = time.time()
        loaded_images = []
        for i, (pic0, pic1) in enumerate(zip(images_camera_0, images_camera_1)):

            loaded_pic_0 = cv2.imread(pic0, cv2.IMREAD_UNCHANGED)
            loaded_pic_1 = cv2.imread(pic1, cv2.IMREAD_UNCHANGED)
            vehicles = self.getRealVehiclePositions(i, vehiclePositions)

            loaded_images.append(StereoVisionImage(loaded_pic_0, loaded_pic_1, vehicles))

            # convert RGB between BGR : cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sys.stdout.write("{}s\n".format(time.time() - start_time))

        sys.stdout.write("Searching for Vehicles in {} images...".format(len(loaded_images)))
        start_time = time.time()
        # TODO: implement
        sys.stdout.write("{}s\n".format(time.time() - start_time))


        print("Show Range estimations...\n")

        for stereo_vision_image in loaded_images:
            img0 = stereo_vision_image.image0
            img1 = stereo_vision_image.image1
            real_vehicle_positions = stereo_vision_image.real_vehicle_positions

            if not plt.get_fignums():
                # window has been closed
                return

            #print ('new Image:')
            ax1=fig.add_subplot(211)
            ax1.imshow(img0,cmap='gray')


            ax2=fig.add_subplot(212)

            # TODO: use vehicle positions from image analysis
            count = len(real_vehicle_positions)
            for j in range(count):
                v = real_vehicle_positions[j]
                name = v.type
                color = self.getVehicleColor(name)

                # show car in top-view coord system
                ax2.add_patch(
                    patches.Rectangle((- v.yPos + v.width, v.xPos - v.length), v.width, v.length, angle=v.angle,
                                      color=color))

                # show vehicle position in image
                vehicleCoord = [v.xPos, v.yPos, v.zPos]
                distance = util.distance(self.camera_model_1.getCameraPosition(), vehicleCoord)
                image_coords = self.camera_model_velo_camera_0.projectToImage(vehicleCoord)
                patch = patches.Rectangle(image_coords, 20, 20, color=color)
                ax1.add_patch(patch)
                # add distance description
                ax1.text(image_coords[0], image_coords[1]+40, "{}m".format(distance), color=color)


            #fig.draw
            ax2.set_ylim([0,100])
            ax2.set_xlim([-25,25])
            ax2.set_aspect(1)

            if self.yield_frames:
                yield fig
            else:
                plt.pause(0.00000001)

            fig.clear()
            #time.sleep(10)
            i+=1

class StereoVisionImage:
    def __init__(self, imageWithVehicles0, imageWithVehicles1, real_vehicle_positions):
        self.image0 = imageWithVehicles0
        self.image1 = imageWithVehicles1
        self.real_vehicle_positions = real_vehicle_positions
        self.vehicles0 = None
        self.vehicles1 = None

    def setVehicles(self, vehicles0, vehicles1):
        self.vehicles0 = vehicles0
        self.vehicles1 = vehicles1