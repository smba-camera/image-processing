import numpy as np

#A class for simulating weather effects
class Weather(object):
    WEATHER_TYPE = dict(enumerate({"Sunny", "Rain", "Snow", "Hail", "Fog"}))

    def __init__(self, current_weather="Sunny", size=0, density=0):
        self.current_weather = current_weather
        if (current_weather != "Sunny"):
            self.density = density
            if (current_weather != "Fog"):
                self.meteor_size = size

    def getWeatherTypes(self):
        return self.WEATHER_TYPE

    def setHydrometeorValues(self, size, density, streak_angle):
        self.meteor_size = size  # in mm
        self.density = density  # in mm/hour
        self.streak_angle = streak_angle  # [0 to pi] pi radians = 180 degrees

    def setFogDensity(self, density):
        self.density = density

    '''
    ls = length of streak

    '''

    def setSimulatedWeather(self, ls=10, ws=1, mu=45, sigma=10):
        return np.random.normal(mu, sigma, ls * ws)