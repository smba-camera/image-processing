import math
import numpy

def distance(A, B):
    def diff_pow(t):
        a,b = t
        return  (a-b) ** 2
    return math.sqrt(sum(map(diff_pow, zip(A, B))))

class Vector3D:
    def __init__(self, start_point, vector):
        self.start_point = numpy.array(start_point)
        self.vector = numpy.array(vector)

    def closest_point(self, target_point):
        min_step_size = 0.0001
        def calc_next_point(X, step_size):
            return X + (self.vector * step_size)
        # iterative step by step of shortest distance
        def recursive_search(target, X, step_size):
            # todo newton method for speed up
            if math.fabs(step_size ) < min_step_size: return X    # step size has reached minimum
            curr_distance = distance(target, X)
            if curr_distance == 0: return X # directly hit target point
            curr_point = X
            while 1:
                next_point = calc_next_point(curr_point, step_size)
                next_distance = distance(target, next_point)
                if next_distance > curr_distance:
                    # we passed shortest distance - turn around
                    return recursive_search(target, next_point, -step_size/2)
                # next iteration
                curr_point = next_point
                curr_distance = next_distance

        return recursive_search(target_point, self.start_point, 10)

    def shortest_distance(self, target_point):
        p = self.closest_point(target_point)
        return distance(p, target_point)

if __name__ == "__main__":

    v = Vector3D([1,1,1] , [0,10,0])
    target_p = [1,31,3]
    shortest_distance = v.shortest_distance(target_p)
    print (shortest_distance)
    assert(shortest_distance < 3)