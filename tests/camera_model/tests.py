import math
from image_processing.camera_model import CameraModel, ExtrinsicModel, IntrinsicModel
import numpy

def test_projection_to_image():
    cm = CameraModel()
    ic = cm.projectToImage([1,1,1])
    maxDiff = 0.000001
    #print("test_projection_to_image:\n{}\n".format(ic))
    exp_coords = [1.0, 1.0]
    assert(abs(ic[0] - exp_coords[0]) < maxDiff)
    assert(abs(ic[1] - exp_coords[1]) < maxDiff)
    assert(len(ic) == 2)

def test_projection_to_image_and_back():
    cm = CameraModel()
    real_coords = [10, 10, 10]
    #print("\n##################\ntest_projection_to_image_and_back: {}".format(real_coords))

    img_coords = cm.projectToImage(real_coords)
    real_coord_vector = cm.projectToWorld(img_coords)

    distance_from_real_coords = real_coord_vector.shortest_distance(real_coords)
    closest_coord = real_coord_vector.closest_point(real_coords)
    #print("Closest point: \n{}\nSmallest Distance: \n{}\n".format(closest_coord, distance_from_real_coords))
    assert(distance_from_real_coords < 1)

def test_projectionToImageAndBack_with_translation():
    e1 = ExtrinsicModel(
        translationVector=[1,3,4])
    extrinsic_models = [e1]
    cm = CameraModel(em=extrinsic_models)
    real_coords = [70, 30, -70]
    #print("\n##################\ntest_projectionToImageAndBack_with_translation: {}".format(real_coords))

    img_coords = cm.projectToImage(real_coords)
    real_coord_vector = cm.projectToWorld(img_coords)

    distance_from_real_coords = real_coord_vector.shortest_distance(real_coords)
    closest_coord = real_coord_vector.closest_point(real_coords)
    #print("Closest point: \n{}\nSmallest Distance: \n{}\n".format(closest_coord, distance_from_real_coords))
    expected_max_distance = 4
    assert(distance_from_real_coords < expected_max_distance)

def test_projectionToImageAndBack_with_rotated_extrinsicModel():
    e1 = ExtrinsicModel(
        rotation=[math.pi/2, math.pi, 0],
        translationVector=[1,3,4])
    extrinsic_models = [e1]
    cm = CameraModel(em=extrinsic_models)
    real_coords = [70, 30, -70]
    #print("\n##################\ntest_projectionToImageAndBack_with_rotated_extrinsicModel: {}".format(real_coords))

    img_coords = cm.projectToImage(real_coords)
    real_coord_vector = cm.projectToWorld(img_coords)

    distance_from_real_coords = real_coord_vector.shortest_distance(real_coords)
    closest_coord = real_coord_vector.closest_point(real_coords)
    #print("Closest point: \n{}\nSmallest Distance: \n{}\n".format(closest_coord, distance_from_real_coords))
    expected_max_distance = 4
    assert(distance_from_real_coords < expected_max_distance)

def test_projectionToImageAndBack_with_multiple_extrinsicModels():
    e1 = ExtrinsicModel(
        rotation=[math.pi/2, math.pi, 0],
        translationVector=[1,3,4])
    e2 = ExtrinsicModel(
        rotation=[math.pi*2, 0, math.pi],
        translationVector=[7,8,9])
    extrinsic_models = [e1, e2]
    cm = CameraModel(em=extrinsic_models)
    real_coords = [10, 10, 10]
    #print("\n##################\ntest_projectionToImageAndBack_with_multiple_extrinsicModels: {}".format(real_coords))

    img_coords = cm.projectToImage(real_coords)
    real_coord_vector = cm.projectToWorld(img_coords)

    distance_from_real_coords = real_coord_vector.shortest_distance(real_coords)
    closest_coord = real_coord_vector.closest_point(real_coords)
    #print("Closest point: \n{}\nSmallest Distance: \n{}\n".format(closest_coord, distance_from_real_coords))
    expected_max_distance = 1
    success = distance_from_real_coords < expected_max_distance
    print("test_projectionToImageAndBack_with_multiple_extrinsicModels:\nDistanceFromRealCoords: {}\nExpectedDistance:{}\n"
          .format(distance_from_real_coords, expected_max_distance))
    assert(success)

def test_roation_from_direction():
    direction = [2,3,-4]
    e = ExtrinsicModel(direction=direction)
    rot_mat = e.getRotationMatrix()
    coords = [5,5,5]
    projected_coords = numpy.matmul(rot_mat, direction)
    exp_length = numpy.linalg.norm(coords)
    length =  numpy.linalg.norm(projected_coords)
    #print("\nLength 1: {}, Length2: {}".format(exp_length, length))
    #print(projected_coords)
    #assert(length == exp_length)
    assert((projected_coords == [1,1,1]).all())