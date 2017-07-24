import numpy
import math
import numbers
import image_processing.util as util


class ExtrinsicModel:

    def __init__(self, rotation=None, direction=None, translationVector=None, position=None):
        """Params: [rotation=rotationMatrix or rotationVector]"""
        if rotation:
            assert(len(rotation) == 3)
            if (isinstance(rotation[0], numbers.Number)):
                # rotation vector defines rotations
                self.alpha = rotation[0]
                self.beta = rotation[1]
                self.gamma = rotation[2]
            else:
                assert(len(rotation[0]) == 3)
                # rotation is defined by given matrix
                self.rotationMatrix = numpy.matrix(rotation)
                assert(self.rotationMatrix.shape[0] == 3 and self.rotationMatrix.shape[1] == 3)
        else:
            if direction:
                def unzero(val):
                    if val==0:
                        return 0.1
                    return val
                normed_direction = util.norm([unzero(x) for x in direction])
                # rotation is given as direction
                self.rotationMatrix = numpy.linalg.inv(
                    [
                        [normed_direction[0],0,0],
                        [0,normed_direction[1],0],
                        [0,0,normed_direction[2]]
                    ])
            else:
                self.alpha = 0 # rotation of x axis
                self.beta = 0 # rotation of y axis
                self.gamma = 0 # rotation of z axis
        if translationVector:
            assert(len(translationVector) == 3)
            assert(all([numpy.isscalar(x) for x in translationVector]))
            self.translation_x = translationVector[0]  # camera position in real world
            self.translation_y = translationVector[1]
            self.translation_z = translationVector[2]
        elif position:
            assert (len(position) == 3)

            translation = numpy.array(numpy.matmul(self.getRotationMatrix(), position))
            #print("positiontranslation: {}\n".format(translation))
            self.translation_x = translation[0][0]
            self.translation_y = translation[0][1]
            self.translation_z = translation[0][2]
        else:
            self.translation_x = 0  # camera position in real world
            self.translation_y = 0
            self.translation_z = 0

    def nonzero(self, num):
        def nonzero_val(val):
            if val==0: return 0.0000000000000001
            return val

        # for rotation non zero values are important
        if not type(num)=='list': return nonzero_val(num)
        return [nonzero_val(x) for x in num]

    def project_coordinates(self, coord):
        assert(len(coord) == 3)

        nonzero_coords = self.nonzero(coord)
        coord_mat = numpy.matrix(nonzero_coords).transpose()
        result = numpy.matmul(self.getRotationMatrix(), coord_mat) + self.getTranslationVector()
        return numpy.array(result).flatten().tolist() # return as normal list

    def project_coordinates_backwards(self, coord):
        assert(len(coord) == 3)

        nonzero_coords = self.nonzero(coord)
        coord_mat = numpy.matrix(nonzero_coords).transpose()
        coord_mat -= self.getTranslationVector()
        rot_mat_inv = numpy.linalg.inv(self.getRotationMatrix())
        result = numpy.matmul(rot_mat_inv, coord_mat)
        return numpy.array(result).flatten().tolist() # return as normal list

    def getRotationMatrix(self):

        if (hasattr(self, 'rotationMatrix')):
            return self.rotationMatrix

        cosA = math.cos(self.alpha)
        sinA = math.sin(self.alpha)
        Rx = numpy.matrix([
            [1 , 0    , 0],
            [0 , cosA , - sinA],
            [0 , sinA , cosA]
        ])
        cosB = math.cos(self.beta)
        sinB = math.sin(self.beta)
        Ry = numpy.matrix([
            [cosB , 0 , sinB],
            [0    , 1 , 0],
            [-sinB, 0 , cosB],
        ])
        cosC = math.cos(self.gamma)
        sinC = math.sin(self.gamma)
        Rz = numpy.matrix([
            [cosC , -sinC , 0],
            [sinC , cosC , 0],
            [0    , 0    , 1]
        ])

        return numpy.matmul(Rx, numpy.matmul(Ry, Rz))

    def getTranslationVector(self):
        return numpy.matrix([self.translation_x, self.translation_y, self.translation_z]).transpose()

    def getTranslation(self):
        return [self.translation_x, self.translation_y, self.translation_z]

    def getMatrix(self):
        R = self.getRotationMatrix()
        t = self.getTranslationVector()
        return numpy.concatenate((R, t), axis=1)

    def getDirection(self):
        vector = numpy.matrix([1,1,1]).transpose()
        rot_mat_inv = numpy.linalg.inv(self.getMatrix())
        result = numpy.matmul(rot_mat_inv, vector)
        return numpy.array(result.transpose()).tolist()[0] # convert to normal python list


