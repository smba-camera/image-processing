import math
import image_processing.camera_model.Models as Models

def test_projection_to_image():
    cm = Models.CameraModel()
    ic = cm.projectToImage([0,0,0])
    assert(ic[0] == 0)
    assert(ic[1] == 0)
    assert(len(ic) == 2)

def test_projection_to_image_and_back():
    cm = Models.CameraModel()
    real_coords = [10, 10, 10]
    img_coords = cm.projectToImage(real_coords)
    real_coord_vector = cm.projectToWorld(img_coords)

    distance_from_real_coords = real_coord_vector.shortest_distance(real_coords)
    assert(distance_from_real_coords < 1)

def test_projectionToImageAndBack_with_rotated_extrinsicModel():
    e1 = Models.ExtrinsicModel(
        rotation=[math.pi/2, math.pi, 0],
        translationVector=[1,3,4])
    extrinsic_models = [e1]
    cm = Models.CameraModel(em=extrinsic_models)

    real_coords = [70, 30, -70]
    img_coords = cm.projectToImage(real_coords)
    real_coord_vector = cm.projectToWorld(img_coords)

    distance_from_real_coords = real_coord_vector.shortest_distance(real_coords)
    expected_max_distance = 4
    assert(distance_from_real_coords < expected_max_distance)

def test_projectionToImageAndBack_with_multiple_extrinsicModels():
    e1 = Models.ExtrinsicModel(
        rotation=[math.pi/2, math.pi, 0],
        translationVector=[1,3,4])
    e2 = Models.ExtrinsicModel(
        rotation=[math.pi*2, 0, math.pi],
        translationVector=[7,8,9])
    extrinsic_models = [e1, e2]
    cm = Models.CameraModel(em=extrinsic_models)

    real_coords = [10, 10, 10]
    img_coords = cm.projectToImage(real_coords)
    real_coord_vector = cm.projectToWorld(img_coords)

    distance_from_real_coords = real_coord_vector.shortest_distance(real_coords)
    expected_max_distance = 4
    assert(distance_from_real_coords < expected_max_distance)
