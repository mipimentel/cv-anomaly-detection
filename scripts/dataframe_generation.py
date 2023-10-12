import glob
import os
import shelve

import cv2 as cv
import imagehash
import pandas as pd
from PIL import Image

from cv_anomaly_detection.img_dataframe import ImageDataFrameMVTEC
from cv_anomaly_detection.metrics import img_metrics
from cv_anomaly_detection.utils import MVTEC_AD

if __name__ == "__main__":
    img_df = ImageDataFrameMVTEC(category="cable")
    img_df.init_default()
