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

def match_2d_coordinate_partners(l1, l2, alpha=40):
    l2_c = list(l2)
    def find_partner(c):
        possible_partners = [x for x in l2_c if distance(x,c) < 40]
        if not possible_partners:
            return None
        p = sorted(possible_partners)[0]
        l2_c.remove(p)
        return p
    result = []
    for c1 in l1:
        p = find_partner(c1)
        if p:
            result.append((c1,p))
        else:
            result.append((c1,None))
    for c2 in l2_c:
        result.append((None, c2))
    return result