import numpy
import image_processing.camera_model

class PositionEstimationStereoVision():
    def __init__(self, camera_model_one, camera_model_two):
        self.camera_model_one = camera_model_one
        self.camera_model_two = camera_model_two

    def estimate_position(self, pos_img_one, pos_image_two):
        vect_one = self.camera_model_one.projectToWorld(pos_img_one)
        vect_two = self.camera_model_two.projectToWorld(pos_image_two)

        #print("\nvect1: \n{}\nVect2: \n{}\n".format(vect_one.vector, vect_two.vector))
        p1, p2 = vect_one.closest_points_to_line(vect_two)
        vect = p1-p2
        vect *= 0.5
        point_in_between = p1 + vect
        return point_in_between

    def estimate_range_stereo(self, pos_img_one, pos_image_two):
        estimated_pos = self.estimate_position(pos_img_one, pos_image_two)
        range_vect = estimated_pos - self.camera_model_one.getCameraPosition()
        return numpy.linalg.norm(range_vect)
