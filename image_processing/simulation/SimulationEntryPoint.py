from image_processing.simulation import Simulation,Weather
from image_processing.hidrometeor_detection import WeatherInfluence

if __name__ == '__main__':
    print("running application")
    #put anything for running standalone here
    Simulation.startSimulation()
    #percent = WeatherInfluence.getPercentageBasedOnSimulatedPixels(image_width=2*640,image_height=2*320,weather=Weather("Hail",20,100))
    #print("affected portion of image: "+str(percent))
    print("end of application execution")

#put anything for the integration to the simulation here