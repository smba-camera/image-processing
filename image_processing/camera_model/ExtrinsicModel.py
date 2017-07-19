import numpy
import math
import numbers
from .IntrinsicModel import IntrinsicModel


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
                        return 0.00000001
                    return val
                # rotation is given as direction
                self.rotationMatrix = numpy.linalg.inv(
                    [
                        [unzero(direction[0]),0,0],
                        [0,unzero(direction[1]),0],
                        [0,0,unzero(direction[2])]
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

    def getMatrix(self):
        R = self.getRotationMatrix()
        t = self.getTranslationVector()
        return numpy.concatenate((R, t), axis=1)