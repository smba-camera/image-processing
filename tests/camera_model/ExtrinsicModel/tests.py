import numpy
from image_processing.util import norm
from image_processing.camera_model import ExtrinsicModel

def test_backwards_projection_projects_translation_to_zero():
    em = ExtrinsicModel(translationVector=[1,3,5], direction=[4,3,5])
    result = em.project_coordinates_backwards([1,3,5])
    assert([0,0,0] == result) # translation in projected coordinates results in [0,0,0]

def test_projection_coordinates_in_direction_become_one():
    direction = [1,0.00,-19]
    em = ExtrinsicModel(direction=direction)
    result = em.project_coordinates(direction)
    result = norm(result)
    expected = norm([1,1,1])
    diff = numpy.subtract(expected, result)
    assertion_successful = all([val < 0.00001 for val in diff])
    if not assertion_successful:
        print("test_projection_coordinates_in_direction_become_one: result:{}".format(result))
    assert(assertion_successful)
