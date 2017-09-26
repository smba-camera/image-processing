import random as rand
import numpy as np

from Car import *
from image_processing.johnsons_criteria import *


'''
A 2D (XY axis) representation of the real world where the ego car is at the origin (0,0) and facing towards the X axis
and all other cars are simulated at different locations within this playground
it also holds a weather object that simulates the weather
'''
class Playground(object):

    def __init__(self,weather,size_x=500,size_y=500, otherCarsNo=3,johnsons_criteria_goal="identification"):
        self.cars_number = otherCarsNo
        self.weather = weather
        self.x = size_x
        self.y = size_y
        self.simCars = [Car]*self.cars_number
        self.jc = JohnsonCriteria()
        self.jc_goal = johnsons_criteria_goal

    def changeJcGoal(self,new_goal):
        self.jc_goal = new_goal

    def addEgoCar(self,my_car):
        self.egoCar = my_car

    def moveOtherCars(self):
        for i in range(0, self.cars_number):
            self.simCars[i].moveCar(rand.randint(0,self.x), rand.randint(-self.y/2,self.y/2),rand.uniform(0,np.pi))

    def changeWeather(self, weather):
        self.weather = weather

    def addRandomCars(self):
        for i in range(0, self.cars_number):
            self.simCars[i] = Car(rand.randint(0,self.x), rand.randint(-self.y/2,self.y/2), 5, 2, rand.uniform(0,np.pi))

    def getNoOfCars(self):
        return self.cars_number

    def getDetectedObjects(self):
        detectedList = [False]*self.cars_number
        for i in range(0, self.cars_number):
            detectedList[i] = self.jc.isCarDetected(goal=self.jc_goal,im=self.egoCar.camera.getIntrinsicModel(),car=self.simCars[i],weather=self.weather)

        return detectedList
