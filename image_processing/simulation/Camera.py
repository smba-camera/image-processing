class Camera(object):

    def __init__(self,resolution_height,resolution_width,lens_angle):
        self.resolution_height = resolution_height
        self.resolution_width= resolution_width
        self.lens_angle = lens_angle

    def convertLPtoPixel(self,linePairs):
        return linePairs*2

    def getFieldOfView(self,goal="Identification"):
        return