import image_processing.position_estimation
import image_processing.camera_model
import math

def test_range_estimation():
    cm1 = image_processing.camera_model.CameraModel()

    em2 = image_processing.camera_model.ExtrinsicModel(position=[0.5,0,0])
    cm2 = image_processing.camera_model.CameraModel(em=em2)

    re = image_processing.position_estimation.PositionEstimationStereoVision(cm1, cm2)

    real_coord = [10,10,10]
    img_coord1 = cm1.projectToImage(real_coord)
    img_coord2 = cm2.projectToImage(real_coord)

    estimated_range = re.estimate_range_stereo(img_coord1,img_coord2)
    expected_range = 17.3205
    success = (math.fabs(estimated_range - expected_range)) < 0.0001
    if not success:
        print("test_range_estimation(): actual_range: {} expected_range: {}".format(estimated_range, expected_range))
    assert(success)
    #print("test_range_estimation: estimated range: {}".format(estimated_range))
