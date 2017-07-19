from image_processing.camera_model import CameraModel, ExtrinsicModel

def test_getCameraAngle():
    expected_direction = [4,5,6]
    em = ExtrinsicModel(direction=expected_direction)
    cm = CameraModel(em=em)

    result = cm.getCameraDirection()
    result -= expected_direction
    assert(all([x<0.0001 for x in result])) # check if expected direction is the result