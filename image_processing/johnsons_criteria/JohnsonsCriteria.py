import numpy as np
import math
from image_processing.simulation import Car, Weather
from image_processing.camera_model import IntrinsicModel
from image_processing.hidrometeor_detection import WeatherInfluence

class JohnsonCriteria:

    # pixels on target according to the Johnson's Criteria with tolerance
    # 50% probability of an observer discriminating an object to the specified level
    johnsons_criteria = [{"detection":{"pixels":2,"tolerance":0.5}},
                            {"orientation":{"pixels":2.8,"tolerance":0.7}},
                            {"recognition":{"pixels":8,"tolerance":1.6}},
                            {"identification":{"pixels":12.8,"tolerance":3}}]

    # another one from: https://www.axis.com/files/feature_articles/ar_perfect_pixel_count_55971_en_1402_lo.pdf
    car_pixels_per_m_2 = {"detection":25,"recognition":125,"identification":250,"id_challenging":500}

    # a car needs the below pixels/m information from: http://www.ev3000.com/EV3000IR.html
    car_pixels_per_m = {"detection":math.ceil(johnsons_criteria[0]["detection"]["pixels"]+johnsons_criteria[0]["detection"]["tolerance"]),
                        "recognition":math.ceil(johnsons_criteria[2]["recognition"]["pixels"]+johnsons_criteria[2]["recognition"]["tolerance"]),
                        "identification":math.ceil(johnsons_criteria[3]["identification"]["pixels"]+johnsons_criteria[3]["identification"]["tolerance"])}

    def isInFieldOfView(self,cameraFov,carAngle):
        """
        Checks if the car from the simulation is within the field of view of the camera
        :param cameraFov: lens angle of the camera for the field of view
        :param carAngle: the offset angle of the car from the X axis centered at the ego car. (0 means the car is on the X axis)
        :return: True if the car is in the cameras field of view
        """
        lowCameraDetectionRange = -cameraFov/2
        highCameraDetectionRange = cameraFov/2
        if (carAngle>=lowCameraDetectionRange and carAngle<=highCameraDetectionRange):
            return True
        else: return False

    def getTargetVisibleSize(self,length,width,carRotation):
        """
        Checks how much of the car can be seen by the camera
        :param length: length of car
        :param width: width of car
        :param carRotation: the angle of the cars body with respect to the X axis (0 means the car is parallel with the X axis)
        :return: a number corresponding to how big is the targets visible size from the ego cars camera point of view
        """
        if (carRotation>np.pi/2): carRotation = np.pi - carRotation
        carRotation = np.pi/2 - carRotation

        lengthProjected = length*np.cos(carRotation)
        widthProjected = width*np.cos(np.pi-(np.pi/2+carRotation))

        targetVisibleSize = widthProjected+lengthProjected

        return targetVisibleSize

    #according to https://kintronics.com/calculating-can-see-ip-camera/ + other calculators and common sense
    def getMaximumRange(self,pixelsPerM, horizontalImagePixels, visibleTargetArea, cameraLensFov):
        """
        Finds the maximum range at which the goal is satisfied by the johnsons criteria
        :param pixelsPerM: the goal criteria according to the Johnsons Criteria
        :param horizontalImagePixels: horizontal size of the image in pixels
        :param visibleTargetArea: size in m of the visible target
        :param cameraLensFov: field of view in radians of the camera lens
        :return: the maximum distance at which the goal is satisfied give the specific parameters
        """
        fieldOfView = horizontalImagePixels*visibleTargetArea/pixelsPerM
        fieldOfView = fieldOfView/2
        cameraLensFov = cameraLensFov/2
        tanLens = math.tan(cameraLensFov)
        distance = fieldOfView/tanLens

        #(elektrobit slide 12) d is the lower bound distance to detect the target (too optimistic)
        d = horizontalImagePixels*visibleTargetArea/cameraLensFov

        #print("target",visibleTargetArea,"logic:",distance,"elektrobit",d)
        return distance

    def getDistanceToTarget(self,target_x,target_y,ego_x=0,ego_y=0):
        #a simple euclidian distance between ego car and target car
        return math.hypot(target_x - ego_x, target_y - ego_y)

    def isCarDetected(self,
                      goal="detection",
                      im = IntrinsicModel(focal_length=1,optical_center_x=0,optical_center_y=0,
                            ratio_image_coordinate_x=1,ratio_image_coordinate_y=1,pixel_skew=0,
                            fov_horizontal=np.pi/6,fov_vertical=np.pi/6,
                            pixel_size=1,image_width=640,image_height=480),
                      car=Car(x=20, y=15, length=4.7, width=1.9, theta=90),
                      weather=None
                      ):
        """
        Checks if the target (car) is detected by by the ego car or not
        :param goal: the goal according to the johnsons criteria: detection, recognition or identification
        :param im: the intrinsic model of the specific camera model being tested
        :param car: the target care being tested against
        :param weather: the specific weather influencing the test
        :return: true if the car is detected and false if not
        """
        car_length = car.length  # m
        car_width = car.width  # m
        car_rotation = car.theta # degrees from Y axis 0 = 180 = horizontal, 90 = vertical
        car_position = [car.x_coor,car.y_coor] #x, y

        #needed only if the ego car position is not at (0,0)
        camera_position = [im.optical_center_x,im.optical_center_y]

        carAngle = math.atan2(car_position[1], car_position[0]) #returns the angle between the x axis and the car position in radians

        if (self.isInFieldOfView(im.fov_horizontal,carAngle)):
            print("tested car is in fov")
            visibleTargetArea = self.getTargetVisibleSize(car_length,car_width,car_rotation)

            rainCoverPercent = WeatherInfluence.getPercentageBasedOnSimulatedPixels(im.image_width,im.image_height,weather)
            # reduce the image size and the car size by the amount of rain percentage covering the image, it assumes rain is uniformly distributed
            nonRainImageWidth = im.image_width - rainCoverPercent * im.image_width
            nonRainVisibleTargetArea = visibleTargetArea - rainCoverPercent * visibleTargetArea

            maxRange = self.getMaximumRange(self.car_pixels_per_m[goal],nonRainImageWidth,nonRainVisibleTargetArea,im.fov_horizontal)
            maxRangeNoWeather = self.getMaximumRange(self.car_pixels_per_m[goal],im.image_width,visibleTargetArea,im.fov_horizontal)

            distanceToTarget = self.getDistanceToTarget(car_position[0],car_position[1])
            print("max range with weather: ",maxRange,"without weather",maxRangeNoWeather,"distance to target",distanceToTarget," goal",goal)
            if (maxRange>distanceToTarget):
                print("tested car is within max range of johnsons criteria")
                return True
            else:
                print("tested car is outside max range of johnsons criteria")
                return False
        else:
            return False