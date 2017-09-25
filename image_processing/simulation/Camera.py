class Camera(object):

    def __init__(self,resolution_width,resolution_height,lens_angle):
        self.resolution_height = resolution_height
        self.resolution_width= resolution_width
        self.lens_angle = lens_angle

    def estimateVisibility(self):
        return  -1

    def convertLPtoPixel(self,linePairs):
        return linePairs*2

    def getFieldOfView(self,goal="Identification"):
        return