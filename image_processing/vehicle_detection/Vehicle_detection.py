from vehicle_detection_svm import init, find_vehicles

class VehicleDetection:
    def __init__(self, img):
        # color image is needed
        height, width, channels = img.shape
        init('svm_model.pkl','scalar.pkl', height, width)
    def find_vehicles(self, img):
        return find_vehicles(img)