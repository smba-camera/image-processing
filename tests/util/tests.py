from image_processing.util import Vector3D,distance

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
    assert(shortest_distance < 3)
