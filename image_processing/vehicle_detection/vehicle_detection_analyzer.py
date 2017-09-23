import os
import glob,sys
import cv2
import numpy as np
import itertools


# make modules accessible for the script
sys.path.append(os.path.abspath(os.path.join(".")))

import image_processing.vehicle_detection.detect_vehicles_serialize as serialize
import image_processing.vehicle_detection.Vehicle_detection as vd
import image_processing.vehicle_detection.stereo_vision_vehicle_matcher as vm
import image_processing.position_estimation.position_estimation as pe
import image_processing.kitti_data.Kitti as kitti
import image_processing.kitti_data.vehicle_positions as vp
import image_processing.util.Util as util

''' uses vehicle detection to mark all vehicles on the kitti images '''
class VehicleDetectionAnalyzer():
    def __init__(self):
        self.matchedCars=[]

    def runComparison(self,date,drive,datapath_left,datapath_right,alpha, distance_steps=10,startFrame=0,maxFrame=0):
        path = os.path.abspath(os.path.join('data', 'kitti'))
        date = '2011_09_26'
        kittiDataLoader=kitti(path,date)
        vehiclePositions=vp.VehiclePositions(path,date,drive)
        leftCameraModel=kittiDataLoader.getCameraModel(3)
        rightCameraModel=kittiDataLoader.getCameraModel(2)
        veloCameraModel=kittiDataLoader.getVeloExtrinsicModel()
        positionEstimator= pe.PositionEstimationStereoVision(leftCameraModel,rightCameraModel)
        #datapath_left='0056_03_0-10_t975'
        #datapath_right = '0056_02_0-10_t975'
        detectedCarCount=0
        realCarCount=0
        carsLeft = serialize.load_detected_vehicles(datapath_left)
        carsRight = serialize.load_detected_vehicles(datapath_right)
        maxFrame = maxFrame if maxFrame else len(carsLeft)
        for framecount in range(startFrame,maxFrame):
            matchedStereoCars = vm.match_vehicles_stereo(carsLeft[framecount],carsRight[framecount])
            carPositions=[]
            for pair in matchedStereoCars:
                if pair[0]!=None and pair[1]!=None:
                    x,y,z=positionEstimator.estimate_position(pair[0], pair[1])
                    temp=[-z,x] #[[list[2],-list[0]]]   #
                    carPositions.append(temp)

            #carPositions.sort(key=lambda tup:np.sqrt(tup[0]*tup[0]+tup[1]*tup[1])
            vehicles=vehiclePositions.getVehiclePosition(framecount)

            cars=[]
            for x in range(len(vehicles)):
                if vehicles[x].type in ('Car'):#,'Van','Truck'):
                    b,y,z=veloCameraModel.project_coordinates([vehicles[x].xPos,vehicles[x].yPos,vehicles[x].zPos])
                    cars.append((z,-b))
            realCarCount+=len(cars)
            detectedCarCount+=len(carPositions)
            #cars.sort(key=lambda tup: np.sqrt(tup[0] * tup[0] + tup[1] * tup[1]))
            self.matchedCars.append(util.match_2d_coordinate_partners(cars,carPositions,alpha))

        self.calculate_detection_error_rate(distance_steps)

    def reset(self):
        self.matchedCars=[]

    def get_matched_cars(self):
        return self.matchedCars

    def calculate_detection_error_rate(self, distance_steps):
        car_position = [0,0]
        index_real = 0
        index_detected = 1

        def normalize_distance(distance):
            return (int(distance / distance_steps) + 1) * distance_steps

        all_matched_cars = list(itertools.chain.from_iterable(self.matchedCars))
        matches_without_false = [x for x in all_matched_cars if x[index_real]]
        matches_false_positives = [x for x in all_matched_cars if x[index_detected] and not x[index_real]]

        matches_per_distance = {}
        for match in matches_without_false:
            #print("real pos: {}".format(match[1]))
            distance_from_car = normalize_distance(util.distance(car_position, match[index_real]))
            if not distance_from_car in matches_per_distance:
                matches_per_distance[distance_from_car] = []
            matches_per_distance[distance_from_car].append(match)

        self.error_rate_per_distance = {}
        self.x_mean_deviation_per_distance = {}
        self.y_mean_deviation_per_distance = {}
        self.x_absolute_deviation_per_distance = {}
        self.y_absolute_deviation_per_distance = {}


        x_mean_deviation_per_distance_aggregated = 0
        y_mean_deviation_per_distance_aggregated = 0
        num_found_cars = 0
        for distance in matches_per_distance:
            matches = matches_per_distance[distance]
            num_found_cars_per_distance = 0
            x_deviation_aggregated = 0
            y_deviation_aggregated = 0
            x_mean_deviation_per_distance = 0
            y_mean_deviation_per_distance = 0
            for match in matches:
                if not match[index_detected]:
                    # was not found
                    continue
                num_found_cars += 1
                num_found_cars_per_distance += 1
                distance_from_car = util.distance(car_position, match[index_real])
                # calculate deviations
                # absolute
                x_deviation = abs(match[index_detected][0]-match[index_real][0])
                y_deviation = abs(match[index_detected][1]-match[index_real][1])
                x_deviation_aggregated += x_deviation
                y_deviation_aggregated += y_deviation
                # relative to distance
                x_mean_deviation_per_distance += x_deviation / float(distance_from_car)
                y_mean_deviation_per_distance += y_deviation / float(distance_from_car)
            self.error_rate_per_distance[distance] = num_found_cars_per_distance / float(len(matches))
            if (num_found_cars_per_distance):
                self.x_absolute_deviation_per_distance[distance] = x_deviation_aggregated  / float(num_found_cars_per_distance)
                self.y_absolute_deviation_per_distance[distance] = y_deviation_aggregated / float(num_found_cars_per_distance)
                x_mean_deviation_per_distance_aggregated += x_mean_deviation_per_distance
                y_mean_deviation_per_distance_aggregated += y_mean_deviation_per_distance
                self.x_mean_deviation_per_distance[distance] = x_mean_deviation_per_distance / float(num_found_cars_per_distance)
                self.y_mean_deviation_per_distance[distance] = y_mean_deviation_per_distance / float(num_found_cars_per_distance)
        self.detection_rate = num_found_cars / float(len(matches_without_false))
        self.num_false_negatives = len(matches_without_false) - num_found_cars
        if num_found_cars:
            self.x_mean_deviation = x_mean_deviation_per_distance_aggregated / float(num_found_cars)
            self.y_mean_deviation = y_mean_deviation_per_distance_aggregated / float(num_found_cars)
        else:

            self.x_mean_deviation=None
            self.y_mean_deviation=None
        self.num_false_positives = len(matches_false_positives)

    def get_x_error_rate(self):

        pass


    def get_overall_error_rate(self):
        pass
