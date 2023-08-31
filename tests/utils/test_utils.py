from cv_anomaly_detection.utils import *


def test_paths():
    common_path = os.path.commonpath([BASE_DIR, DATA_DIR, MVTEC_AD])
    # asserts base dir is common path to all child data directories
    assert common_path == BASE_DIR
