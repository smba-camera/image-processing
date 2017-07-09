from random import randint as rand
from Car import *


class Playground(object):
    cars = []

    def __init__(self, x_size, y_size, cars_no, weather):
        self.x = x_size
        self.y = y_size
        self.cars_number = cars_no
        self.weather = weather

    def addMyCar(self,my_car):
        self.my_car = my_car

    def addCarsToPlayground(self,cars):
        self.cars = cars

    def addWeatherToPlaygounr(self,weather):
        self.weather = weather

    def moveMyCar(self):
        self.my_car.moveCar(1,1,75)

    def moveOtherCars(self):
        for i in range(0, self.cars_number):
            self.cars[i].moveCar(1,1,75)

    def changeWeather(self, weather):
        self.weather = weather

    def addRandomCars(self):
        for i in range(0, self.cars_number):
            self.cars[i] = Car(rand(0,self.x), rand(0,self.y), rand(1,5), 1, rand(0,180))
