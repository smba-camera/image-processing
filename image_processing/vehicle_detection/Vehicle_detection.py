from vehicle_detection_svm import init, find_vehicles,show_vehicles,predict64by64image
import os,glob


class VehicleDetection:
    def __init__(self, img):
        # color image is needed
        height, width, channels = img.shape
        init(os.path.join('..','image_processing','vehicle_detection','svm_model.pkl'),os.path.join('..','image_processing','vehicle_detection','scalar.pkl'), height, width)
    def find_vehicles(self, img):
        return find_vehicles(img)
    def show_vehicles(self, img):
        show_vehicles(img)
    def predict(selfself,img):
        return predict64by64image(img)