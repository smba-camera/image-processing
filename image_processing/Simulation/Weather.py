class Weather(object):

    weather_type = {1:"Sunny",
                    2:"Rain",
                    3:"Snow",
                    4:"Hail",
                    5:"Fog"}

    def __init__(self,weather_value):
        self.weather_value = weather_value

    def setHydrometeorValues(self,size,density):
        self.meteor_size = size
        self.density = density

    def setFogDensity(self,density):
        self.density = density
