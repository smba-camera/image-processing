
import os,glob


class VehicleDetection:
    def __init__(self, img):
        # color image is needed
        height, width, channels = img.shape
        from vehicle_detection_svm import init
        init(os.path.abspath(os.path.join('image_processing','vehicle_detection','svm_model.pkl')),os.path.abspath(os.path.join('image_processing','vehicle_detection','scalar.pkl')), height, width)

    def find_vehicles(self, img):
        from vehicle_detection_svm import find_vehicles
        return find_vehicles(img)
    def show_vehicles(self, img):
        from vehicle_detection_svm import show_vehicles
        show_vehicles(img)
    def predict(selfself,img):
        from vehicle_detection_svm import predict64by64image
        return predict64by64image(img)