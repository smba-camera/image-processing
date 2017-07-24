import numpy
import math
from image_processing.util import Vector3D,distance,norm,turnVect_firstTry
from image_processing.camera_model import ExtrinsicModel

def test_distance():
    A, B = [1,3,7], [1,3, 707]
    assert (distance(A, B) == 700)

def test_Distance():
    A = [1,2,3]
    B = [1,10,3]
    assert(distance(A, B) == 8)

def test_shortestDistance():
    v = Vector3D([1,1,1] , [0,10,0])
    target_p = [1,31,3]
    shortest_distance = v.shortest_distance(target_p)
    #print("test_shortestDistance: {}".format(shortest_distance))
    assert(shortest_distance < 3)

def test_distanceToLine():
    u = Vector3D([0,0,0], [3,0,0])
    v = Vector3D([0,2,0], [0,0,3])
    dist = u.distance_to_line(v)
    exp_dist = 2
    assert(dist == exp_dist)

def test_norm():
    vect = [6,7,8]
    normed = norm(vect)
    assert(1 == numpy.linalg.norm(normed))

def test_turn():
    vect = [4,6,8]
    rad_turn = math.pi / 8
    turned = turnVect_firstTry(vect, rad_turn)
    turned_turned = turnVect_firstTry(turned, -rad_turn)
    print("vect: {}".format(vect))
    print("turned_turned: {}".format(turned_turned))
    assert(vect == turned_turned)
