from image_processing.johnsons_criteria import JohnsonsCriteria
from image_processing.camera_model import IntrinsicModel
from image_processing.simulation import Car
class RosIntegration:
    def __init__(self):
        self.a = 1

    def isTargetVisible(self,x,y,heading,width,length):
        car = Car(x=x,y=y,length=length,width=width,theta=heading)
        return JohnsonsCriteria.isCarDetected(car)