from Playground import *
from Car import *
from Weather import *
from image_processing.camera_model import *
import random as rand
import time
import numpy as np

def startSimulation():
    weather = Weather("Sunny",0,0)

    noOfSimCars = 30

    playground = Playground(weather,600,600,noOfSimCars)
    playground.addRandomCars()

    myCar = Car(0,0,5,2,0)
    im = IntrinsicModel(fov_horizontal=np.pi/6,fov_vertical=np.pi/6,image_width=640,image_height=480)
    camera = CameraModel(im=im,em=None,prepare_projection_matrix=False)
    myCar.addCameraToEgoCar(camera)
    playground.addEgoCar(myCar)

    i=1
    print("starting simulation with weather ",weather.current_weather)
    while (i<=100):

        visibilityRange = playground.estimateVisibility()
        detectedObjects = playground.getDetectedObjects(visibilityRange)
        print("cars detected: ", detectedObjects)

        playground.moveOtherCars()
        print("movement simulated")
        #visibilityRange = playground.estimateVisibility()
        #detectedObjects = playground.getDetectedObjects(visibilityRange)
        #print("cars detected: ", detectedObjects)

        if(1==rand.randint(1,10)):
            weather_type = Weather.WEATHER_TYPE[rand.randint(0,4)]
            if(weather_type=="Rain"):
                hydrometeorSize = rand.uniform(0.5,5)
                density = rand.uniform(1,300)
            elif(weather_type=="Hail"):
                hydrometeorSize = rand.uniform(10,40)
                density = rand.uniform(1,200)
            elif(weather_type=="Snow"):
                hydrometeorSize = rand.uniform(0.5,6)
                density = rand.uniform(20,300)
            elif(weather_type=="Fog"):
                hydrometeorSize = 0
                density = rand.uniform(1,300)
            else:
                hydrometeorSize = 0
                density = 0

            weather = Weather(current_weather=weather_type,size=hydrometeorSize,density=density)
            playground.changeWeather(weather)
            print("---------------------------------------------------------------------------------------------weather has changed to ",weather.current_weather)

        print("rounds passed {}",i)
        i=i+1
        #time.sleep(1)
        print("egoCar: ",playground.egoCar.x_coor,playground.egoCar.y_coor,playground.egoCar.theta)

        for j in range(0, playground.getNoOfCars()):
            if(detectedObjects[j]):
                print("detected sim car No=",j,"x=",playground.simCars[j].x_coor,"y=",playground.simCars[j].y_coor,"theta=",math.degrees(playground.simCars[j].theta))

    print("simulation finished")