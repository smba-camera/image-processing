import numpy
import math

import image_processing.util
from .IntrinsicModel import IntrinsicModel
from .ExtrinsicModel import ExtrinsicModel

# The Camera Model consists of one intrinsic and at least one extrinsic model
class CameraModel:
    def __init__(self, im=None, em=None, prepare_projection_matrix=True):
        self.prepare_projection_matrix = prepare_projection_matrix
        if (im):
            self.intrinsic_model = im
        else:
            self.intrinsic_model = IntrinsicModel()

        self.apply_new_extrinsic_models(em)

    # precalculate projection matrix for faster projection
    def calculate_projection_matrix(self):
        one_vect_short = numpy.matrix([1,1,1])
        zero_vect = numpy.matrix([0,0,0,1])

        intr_mat = numpy.concatenate((self.intrinsic_model.getMatrix(), one_vect_short.transpose()), axis=1)
        projection_mat = intr_mat
        for e_model in self.extrinsic_models:
            e_mat = numpy.concatenate((e_model.getMatrix(), zero_vect))
            projection_mat = numpy.matmul(projection_mat, e_mat)

        return projection_mat

    # will set the new extrinsic models and (if 'prepare_projection_matrix' is set) precalculate the projection
    def apply_new_extrinsic_models(self, ems):
        self.extrinsic_models = []
        if (ems):
            if (type(ems) is list):
                self.extrinsic_models = ems
            else:
                self.extrinsic_models.append(ems)
        else:
            self.extrinsic_models.append(ExtrinsicModel())
        if self.prepare_projection_matrix:
            self.projection_matrix = self.calculate_projection_matrix()

    def projectToImage(self, coords):
        assert(len(coords) == 3)
        pos_vect = numpy.matrix(coords).transpose()
        one_vect = numpy.matrix([1])

        Ixyz = None
        if (hasattr(self, 'projection_matrix')):
            pos_vect = numpy.concatenate((pos_vect, one_vect))
            Ixyz = numpy.matmul(self.projection_matrix, pos_vect)
        else:
            for m in self.extrinsic_models:
                # append 1 at end because extrinsic matrices also contain translation
                pos_vect = numpy.concatenate((pos_vect, one_vect))
                pos_vect = numpy.matmul(m.getMatrix(), pos_vect)

            Ixyz = numpy.matmul(self.intrinsic_model.getMatrix(), pos_vect)

        xz = Ixyz[0][0]
        yz = Ixyz[1][0]
        z = Ixyz[2][0]
        if (z == 0): return [xz, yz]

        return  [
             float(xz / z),
             float(yz / z)
        ]

    def projectToWorld(self, coords, implementation=0):
        if implementation == 1:
            return self.projectToWorld_translation(coords)
        elif implementation == 2:
            return self.projectToWorld_invertedTranslation(coords)
        elif implementation == 3:
            return self.projectToWorld_invertedProjection(coords)

        return self.projectToWorld_usingUntranslatedVector(coords)


    def projectToWorld_usingUntranslatedVector(self, coords):#projectToWorld_oldstrangeWorking
        #print("\nPROJECT TO WORLD\n")
        # TODO make it work with multiple extrinsic matrices
        assert(len(coords) == 2)
        result = numpy.matrix([coords[0], coords[1], 1]).transpose()
        result = result
        # intrinsic back calculation
        intr_invert = numpy.linalg.inv(self.intrinsic_model.getMatrix())
        result = numpy.matmul(intr_invert, result)
        # extrinsic back calculation
        for m in reversed(self.extrinsic_models):
            rot_inverted = numpy.linalg.inv(m.getRotationMatrix())
            translation_inv = numpy.matmul(rot_inverted, m.getTranslationVector())
            vector = numpy.matmul(rot_inverted, result)
            result = numpy.subtract(vector, translation_inv)

        # print("Result: \n{}\n".format(result))
        # print("Translation: \n{}\n".format(translation_inverted))
        # print("Vector after: \n{}\n".format(vector))

        v = image_processing.util.Vector3D(result, vector)
        v.norm()
        return v

    def projectToWorld_translation(self, coords):#projectToWorld_translation
        #print("\nPROJECT TO WORLD\n")
        # TODO make it work with multiple extrinsic matrices
        assert(len(coords) == 2)
        result = numpy.matrix([coords[0], coords[1], 1]).transpose()
        result = result
        # intrinsic back calculation
        intr_invert = numpy.linalg.inv(self.intrinsic_model.getMatrix())
        result = numpy.matmul(intr_invert, result)
        # extrinsic back calculation
        for m in reversed(self.extrinsic_models):
            rot_inverted = numpy.linalg.inv(m.getRotationMatrix())
            result = numpy.subtract(result, m.getTranslationVector())
            result = numpy.matmul(rot_inverted, result)

        camera_position = numpy.matmul(rot_inverted, self.extrinsic_models[0].getTranslationVector())
        vector = numpy.subtract(result, camera_position)

        # print("Result: \n{}\n".format(result))
        # print("Translation: \n{}\n".format(translation_inverted))
        # print("Vector after: \n{}\n".format(vector))
        v = image_processing.util.Vector3D(camera_position, vector)
        v.norm()
        return v


    def projectToWorld_invertedTranslation(self, coords):
        #print("\nPROJECT TO WORLD\n")
        # TODO make it work with multiple extrinsic matrices
        assert(len(coords) == 2)
        result = numpy.matrix([coords[0], coords[1], 1]).transpose()
        # intrinsic back calculation
        intr_invert = numpy.linalg.inv(self.intrinsic_model.getMatrix())
        result = numpy.matmul(intr_invert, result)
        # extrinsic back calculation
        for m in reversed(self.extrinsic_models):
            rot_inverted = numpy.linalg.inv(m.getRotationMatrix())
            translation_inverted = numpy.matmul(rot_inverted, m.getTranslationVector())
            result = numpy.matmul(rot_inverted, result)
            result = numpy.subtract(result, translation_inverted)

        vector = numpy.subtract(result, translation_inverted)
        #print("Result: \n{}\n".format(result))
        #print("Translation: \n{}\n".format(translation_inverted))
        #print("Vector after: \n{}\n".format(vector))
        v = image_processing.util.Vector3D(translation_inverted, vector)
        v.norm()
        return v

    def projectToWorld_invertedProjection(self, coords): #projectToWorld_invertedProjection
        # Uses projection matrix - tries to solve multiple extrinsic matrix problem
        assert(len(coords) == 2)

        if (hasattr(self, 'projection_matrix')):
            proj_mat = self.projection_matrix
        else:
            proj_mat = self.calculate_projection_matrix()
        proj_mat = numpy.concatenate((proj_mat, numpy.matrix([0,0,0,0.000000001])))
        proj_mat_inv = numpy.linalg.inv(proj_mat)

        img_coord = numpy.matrix([coords[0], coords[1], 1, 1]).transpose()
        real_coord_long = numpy.matmul(proj_mat_inv, img_coord)
        real_coord = numpy.take(real_coord_long, [0,1,2]).transpose()

        start_point = self.extrinsic_models[0].getTranslationVector()
        vector = real_coord - start_point
        v = image_processing.util.Vector3D(start_point, vector)
        v.norm()
        return v

    def getCameraPosition(self):
        vect_trans = self.extrinsic_models[0].getTranslationVector()
        rot_inv = numpy.linalg.inv(self.extrinsic_models[0].getRotationMatrix())
        camera_position = numpy.matmul(rot_inv, vect_trans)
        return camera_position