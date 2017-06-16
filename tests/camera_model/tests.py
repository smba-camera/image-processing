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