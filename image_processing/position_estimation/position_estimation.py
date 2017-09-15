import numpy
import image_processing.camera_model

class PositionEstimationStereoVision():
    def __init__(self, camera_model_one, camera_model_two):
        self.camera_model_one = camera_model_one
        self.camera_model_two = camera_model_two

    def estimate_position(self, pos_img_one, pos_img_two):
        pos_one=(pos_img_one[0][0] + 0.5 * (pos_img_one[1][0] - pos_img_one[0][0]),pos_img_one[0][1] + 0.5 * (pos_img_one[1][1] - pos_img_one[0][1]))
        pos_two = (pos_img_two[0][0] + 0.5 * (pos_img_two[1][0] - pos_img_two[0][0]),pos_img_two[0][1] + 0.5 * (pos_img_two[1][1] - pos_img_two[0][1]))
        vect_one = self.camera_model_one.projectToWorld(pos_one)
        vect_two = self.camera_model_two.projectToWorld(pos_two)

        p1, p2 = vect_one.closest_points_to_line(vect_two)
        vect = numpy.array(p2)-numpy.array(p1)
        vect *= 0.5
        point_in_between = p1 + vect
        return numpy.array(point_in_between).tolist()[0]

    def estimate_range_stereo(self, pos_img_one, pos_image_two):
        estimated_pos = self.estimate_position(pos_img_one, pos_image_two)
        range_vect = estimated_pos - self.camera_model_one.getCameraPosition().transpose()
        return numpy.linalg.norm(range_vect)

    def estimate3DPosition(self,pos_img_one,pos_img_two):
        estimated_pos = self.estimate_position(pos_img_one, pos_img_two)
        range_vect = estimated_pos - self.camera_model_one.getCameraPosition().transpose()
        return range_vect
