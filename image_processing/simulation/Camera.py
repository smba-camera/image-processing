class Camera(object):

    johnsons_criteria = [
        {"Detection":{"line_pairs":1,"tolerance":0.25}},
        {"Orientation":{"line_pairs":1.4,"tolerance":0.35}},
        {"Recognition":{"line_pairs":4,"tolerance":0.8}},
        {"Identification":{"line_pairs":6.4,"tolerance":1.5}}
    ]

    def __init__(self,resolution_height,resolution_width,lens_angle):
        self.resolution_height = resolution_height
        self.resolution_width= resolution_width
        self.lens_angle = lens_angle

    def convertLPtoPixel(self,linePairs):
        return linePairs*2

    def getFieldOfView(self,goal="Identification"):
        return