from Playground import *
from Car import *
from Weather import *
from image_processing.camera_model import *
import random as rand
import time
import numpy as np

def startSimulation():
    """
    A method that instantiates a simulation for the johnsons criteria range estimation with weather effects.
    It creates an instance of an ego Car that contains a CameraModel and an IntrinsicModel
    It creates n random target cars that are to be detected and a starting simulated weather object
    All of the above are added to the Playground object and the simulation controls the playground object where it can change the weather and move the cars
    The simulation also requests from the playground results on weather the target cars are detected by the ego car or not
    The simulation also runs a certain number of iterations that are predefined by the user.
    """
    weather = Weather("Sunny",0,0)

    data = raw_input('Input number of cars in simulation (integer): ')
    noOfSimCars = int(data)

    data = raw_input('Input size of playground (integer): ')
    playground_size = int(data)

    data = raw_input('Input Johnsons criteria goal (D=Detection,R=Recognition,I=Identification): ')
    jc_goal = data
    if(jc_goal.lower().startswith('D')): jc_goal="detection"
    elif(jc_goal.lower().startswith('R')): jc_goal="recognition"
    else:jc_goal="identification"

    playground = Playground(weather,playground_size,playground_size,noOfSimCars,jc_goal)
    print("Playground created")
    playground.addRandomCars()

    data = raw_input('Input camera lens angle (degrees): ')
    camerafov_radians = np.radians(int(data))
    data = raw_input('Input camera resolution width (pixels): ')
    camera_width_pixels = int(data)
    data = raw_input('Input camera resolution height (pixels): ')
    camera_height_pixels = int(data)

    im = IntrinsicModel(fov_horizontal=camerafov_radians,fov_vertical=camerafov_radians,image_width=camera_width_pixels,image_height=camera_height_pixels)
    camera = CameraModel(im=im,em=None,prepare_projection_matrix=False)
    myCar = Car(0,0,5,2,0)

    myCar.addCameraToEgoCar(camera)
    playground.addEgoCar(myCar)

    data = raw_input('Input number of simulation iterations (integer): ')
    sim_iterations = int(data)

    data = raw_input('Does the weather change randomly (Y/N): ')
    if(data.lower()=='y'): isWeatherChanging = True
    else: isWeatherChanging = False

    if(isWeatherChanging):
        data = raw_input('Input probability of weather to change (between 1-100%): ')
        weather_change_percent = int(data)
    else:
        data = raw_input('Input weather type (S=Sunny,R=Rain,H=Hail,S=Snow): ')
        specific_weather_input = data
        if(specific_weather_input.lower().startswith('r')): weather.current_weather = "Rain"
        elif(specific_weather_input.lower().startswith('h')): weather.current_weather="Hail"
        elif(specific_weather_input.lower().startswith('s')): weather.current_weather="Snow"
        #fog not implemented yet
        data = raw_input('Input hydrometeor size (mm,between 0.5-40): ')
        weather.meteor_size = float(data)
        data = raw_input('Input weather effect density (mm/hour,between 1-300): ')
        weather.density = float(data)
    i=1
    print("starting simulation with weather ",weather.current_weather)
    while (i<=sim_iterations):

        if(isWeatherChanging):
            if(weather_change_percent>=rand.randint(1,100)):
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
                print("The weather has changed to ",weather.current_weather)

        print("rounds passed {}",i)
        i=i+1
        #time.sleep(1)
        #print("egoCar: ",playground.egoCar.x_coor,playground.egoCar.y_coor,playground.egoCar.theta)

        detectedObjects = playground.getDetectedObjects()
        print("cars detected: ", detectedObjects)

        for j in range(0, playground.getNoOfCars()):
            if(detectedObjects[j]):
                print("detected sim car No=",j,"x=",playground.simCars[j].x_coor,"y=",playground.simCars[j].y_coor,"theta=",math.degrees(playground.simCars[j].theta))

        playground.moveOtherCars()
        print("playground movement simulated")

    print("simulation finished")