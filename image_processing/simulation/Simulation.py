import Playground
import Car
import Weather
from random import randint as rand

def startSimulation():
    myCar = Car()
    weather = Weather()
    noOfSimCars = 3
    simCars = []
    for i in range(0,noOfSimCars):
        simCars[i] = Car()

    playground = Playground()
    playground.addWeatherToPlayground(weather)
    playground.addCarsToPlayground(simCars)
    playground.addMyCar(myCar)

    playground.calibrateMyCarCamera()

    while (1==0):
        playground.estimateVisibility()
        detectedObjects = playground.getDetectedObjects()

        playground.moveMyCar()

        playground.estimateVisibility()
        detectedObjects = playground.getDetectedObjects()

        playground.moveOtherCars()

        playground.estimateVisibility()
        detectedObjects = playground.getDetectedObjects()

        if(1==rand(1,10)):
            weather = Weather()
            playground.changeWeather(weather)




