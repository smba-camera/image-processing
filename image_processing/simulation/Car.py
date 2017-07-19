from Image import *

class Car:

    #x and y are reflected as the center of the car the car grows 4.7/2 and 1./9 in the respective directions
    def __init__(self,x=2,y=4,length=4.7,width=1.9,theta=90):
        print("Car created")
        self.x_coor = x
        self.y_coor = y
        self.length = length
        self.width = width
        self.theta = theta #is the angle

    def moveCar(self,new_x,new_y,new_theta):
        self.x_coor = new_x
        self.y_coor = new_y
        self.theta = new_theta

