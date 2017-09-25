from Image import *
import numpy as np

class Car:

    #x and y are reflected as the center of the car rear axle
    def __init__(self,x=0,y=0,length=5,width=2,theta=np.pi):
        print("Car created")
        self.x_coor = x
        self.y_coor = y
        self.length = length
        self.width = width
        self.theta = theta #is the angle

    def addCameraToEgoCar(self,camera):
        self.camera = camera

    def moveCar(self,new_x,new_y,new_theta):
        self.x_coor = new_x
        self.y_coor = new_y
        self.theta = new_theta