import math
import numpy


def distance(A, B):
    def diff_pow(t):
        a,b = t
        return  (a-b) ** 2
    return math.sqrt(sum(map(diff_pow, zip(A, B))))

def norm(vect):
    length = numpy.linalg.norm(vect)
    return map(lambda x: x/length, vect)

