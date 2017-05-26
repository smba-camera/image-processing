import image_processing.camera_model.Models as Models

def test_packageAccess():
    cm = Models.CameraModel()
    ic = cm.projectToImage([0,0,0])
    assert(ic[0] == 0)
    assert(ic[1] == 0)
    assert(len(ic) == 2)
