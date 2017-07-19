class Weather(object):

    weather_type = {"Sunny",
                    "Rain",
                    "Snow",
                    "Hail",
                    "Fog"}

    def __init__(self,current_weather,size=0,density=0):
        self.current_weather = current_weather
        if (current_weather!="Sunny"):
            self.density = density
            if (current_weather!="Fog"):
                self.meteor_size = size

    def getWeatherTypes(self):
        return self.weather_type

    def setHydrometeorValues(self,size,density):
        self.meteor_size = size #in mm
        self.density = density # in mm/hour

    def setFogDensity(self,density):
        self.density = density
