import image_processing.johnsons_criteria

def test_johnsons_criteria():
    jc = image_processing.johnsons_criteria
    visibleRange = jc.estimateRange() #in meters
    assert(visibleRange==-1)