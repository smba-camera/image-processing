import math
import numpy
from .Util import *

class Vector3D:
    def __init__(self, start_point, vector):
        self.start_point = numpy.array(start_point)
        self.vector = numpy.array(vector)

    def length(self):
        return distance(self.vector, [0] * len(self.vector))

    def length_factor(self, factor):
        self.vector[0] *= factor
        self.vector[1] *= factor
        self.vector[2] *= factor

    def norm(self):
        length = numpy.linalg.norm(self.vector)
        self.length_factor(1/length)

    def closest_point(self, target_point):
        min_step_size = 0.0001
        normalized_min_step_size = min_step_size / self.length()

        def calc_next_point(X, step_size):
            return X + (self.vector * step_size)
        # iterative step by step of shortest distance
        def recursive_search(target, X, step_size):
            # todo newton method for speed up
            if math.fabs(step_size ) < normalized_min_step_size:
                return X    # step size has reached minimum
            curr_distance = distance(target, X)
            if curr_distance == 0:
                return X # directly hit target point
            curr_point = X
            while 1:
                next_point = calc_next_point(curr_point, step_size)
                next_distance = distance(target, next_point)
                if next_distance > curr_distance:
                    # we passed shortest distance - turn around
                    return recursive_search(target, next_point, -step_size/2.0)
                # next iteration
                curr_point = next_point
                curr_distance = next_distance

        return recursive_search(target_point, self.start_point, 10)

    def closest_points_to_line(self, line):
        u = self.vector.transpose().tolist()[0] # convert from numpy matrix to normal list
        v = line.vector.transpose().tolist()[0]
        w = self.start_point - line.start_point
        a = numpy.dot(u,u)
        b = numpy.dot(u,v)
        c = numpy.dot(v,v)
        d = numpy.dot(u,w)
        e = numpy.dot(u,w)
        D = a*c - b*b

        epsilon = 0.0000001
        if D < epsilon:
            # intersection of both lines?
            sc = 0.0
            tc = d/b if b>c else e/c
        else:
            sc = (b*e - c*d) / D
            tc = (a*e - b*d) / D
        #dP = w + (sc * u) - (tc * v)

        p1 = self.start_point + sc * self.vector
        p2 = line.start_point + tc * v

        return (p1, p2)

        #return numpy.linalg.norm(dP)

    def distance_to_line(self, line):
        points = self.closest_points_to_line(line)
        dist = numpy.linalg.norm(points[0] - points[1])
        return dist

    def shortest_distance(self, target_point):
        p = self.closest_point(target_point)
        return distance(p, target_point)

if __name__ == "__main__":

    v = Vector3D([1,1,1] , [0,10,0])
    target_p = [1,31,3]
    shortest_distance = v.shortest_distance(target_p)
    print (shortest_distance)
    assert(shortest_distance < 3)