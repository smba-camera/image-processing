class Pixel(object):

    def __init__(self,r_value,g_value,b_value,a_value):
        self.r = r_value #red
        self.g = g_value #green
        self.b = b_value #blue
        self.a = a_value #alpha or grayscale
        #all between 0-255

class Image(object):

    pixel_vector = []

    def __init__(self,x_pixels,y_pixels):
        self.x_pixels = x_pixels
        self.y_pixels = y_pixels

    def setPixelValues(self,pixel_vector):
        self.pixel_vector = pixel_vector

    def setPixelValuesBlank(self):
        for x in range(0,self.x_pixels):
            for y in range(0,self.y_pixels):
                self.pixel_vector[x][y] = Pixel(0,0,0,0)