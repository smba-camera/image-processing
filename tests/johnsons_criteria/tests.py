from image_processing.johnsons_criteria import JohnsonsCriteria as jc

def test_johnsons_criteria():
    visibleRange = jc.estimateRange() #in meters
    assert(visibleRange>0)