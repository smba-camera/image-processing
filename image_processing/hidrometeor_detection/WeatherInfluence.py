from __future__ import division
import numpy as np
import cv2

from image_processing.simulation import Weather
from image_processing.camera_model import *
from image_processing.testimage_preprocessor import image_operations

# rain drop size: 0.5 to 5 (hail 10 - 40) (snow 0.5 to 20)mm / density from 0 to 500mm/hour
def getPercentageBasedOnSimulatedPixels(image_width=640,image_height=320,weather = Weather(current_weather="Rain",size=1,density=150),):
    """
    A method that uses testimage_preprocessor.image_operations to simulate
    fake weather given the weather parameters and then counts the amound of
    affected rain pixels to return a percentage of the image covered by adverse weather effects.
    :param image_width: width of image in pixels
    :param image_height: height of image in pixels
    :param weather: the weather effect being simulated
    :return: the percentage of image pixels affected by the weather
    """
    blank_image = np.zeros((image_height,image_width,3), np.uint8)
    weather_type = weather.current_weather
    if (weather_type == "Sunny"): return 0
    #generate fake rain and count the rain pixels in order to see how much of the image is covered
    #if possible add a link between the amount of rain pixels and the weather forcast (density, droplet size)
    widthstreak = 1
    vertical = image_height/100 - np.log(weather.density/100)
    vertical_variance = image_height/100
    horizontal = image_width/100 - np.log(weather.density/100)
    horizontal_variance = image_width/100
    if (weather_type == "Rain"):
        if(weather.meteor_size>1):
            widthstreak = weather.meteor_size*(1/3)
        else: widthstreak = weather.meteor_size
    elif (weather_type == "Hail"):
        weather.meteor_size = weather.meteor_size*(1/2)
        widthstreak = weather.meteor_size*(1/5)
    elif (weather_type == "Snow"):
        if(weather.meteor_size>1):
            widthstreak = weather.meteor_size*(1/10)
        else: widthstreak = weather.meteor_size
    elif (weather_type == "Fog"):
        widthstreak = 0.001
        weather.meteor_size = 0.01

    fake_rain = image_operations.simulate_rain_by_gaussian(blank_image,vertical=vertical,vertical_variance=vertical_variance,
                                                           horizontal=horizontal,horizontal_variance=horizontal_variance,
                                                           lenstreaks=weather.meteor_size,variance_lenstreaks=1,
                                                           widthstreak=widthstreak,variance_widthstreak=0.5,
                                                           angle=20,variance_angle=8,
                                                           color=(255,255,255),color_variance=0)

    #BLACK = np.array([0, 0, 0], np.uint8)
    WHITE = np.array([255,255,255],np.uint8)

    dst = cv2.inRange(fake_rain, WHITE, WHITE)
    only_white = cv2.countNonZero(dst)
    print('The number of white pixels is: ' + str(only_white)+
          ' percentage in image: '+str(only_white/(image_width*image_height)))
    #cv2.namedWindow("opencv")
    #cv2.imshow("opencv", fake_rain)
    #cv2.waitKey(0)
    return only_white/(image_width*image_height)

def getPercentageBasedOnWeatherInfluence(image_width,image_height,weather = Weather("Rain",size=1,density=150)):
    #an uncompleted method due to lack of information that aims to map direct weather parameters into amound of affected pixels
    weather_type = weather.weather_type
    if(weather.density==0): return 0
    if (weather_type == "Rain"):
        hydrometeorSize = weather.meteor_size
        density = weather.density
        return ((hydrometeorSize-0.5)*density)/(4.5*500+(4.5*500)*(50/100))
    elif (weather_type == "Hail"):
        hydrometeorSize = weather.meteor_size
        density = weather.density
        return ((hydrometeorSize-10)*density)/(30*500+(30*500)*(50/100))
    elif (weather_type == "Snow"):
        hydrometeorSize = weather.meteor_size
        density = weather.density
        return ((hydrometeorSize-0.5)*density)/(19.5*500+(19.5*500)*(50/100))
    elif (weather_type == "Fog"):
        hydrometeorSize = 0
        density = weather.density
        if(density==0): return 0
        return density/(500+500*(20/100))
    else: #the sunny case
        hydrometeorSize = 0
        density = 0
        return 0
    return 0