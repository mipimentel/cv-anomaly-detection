import os

os.environ["KERAS_BACKEND"] = "torch"
import h5py
import keras
import numpy as np
from keras.applications.mobilenet import preprocess_input
from keras.models import Model
from keras.preprocessing import image
from keras.utils import plot_model
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf

from cv_anomaly_detection.img_dataframe import ImageDataFrameMVTEC
from cv_anomaly_detection.utils import DATA_DIR, MVTEC_AD

mnetv3 = keras.applications.MobileNetV3Large(
    input_shape=(224, 224, 3),
    alpha=1.0,
    minimalistic=False,
    include_top=False,
    weights="imagenet",
)

img_df = ImageDataFrameMVTEC(category="cable")
img_df.init_default()

test_df = img_df.df[img_df.df["dataset_type"] == "test"]
train_df = img_df.df[img_df.df["dataset_type"] == "train"]

# Define a model that retrieves outputs from specific layers
layer_outputs = [
    mnetv3.get_layer("activation").output,
    mnetv3.get_layer("activation_10").output,
    mnetv3.get_layer("activation_19").output,
]

feature_extraction_model = Model(inputs=mnetv3.input, outputs=layer_outputs)


def mahalanobis_threshold(k: int, p: float = 0.9973) -> float:
    """
    Computes the Mahalanobis distance threshold for the given number of features and significance level.

    Parameters:
        k (int): The number of degrees of freedom.
        p (float): The significance level. Default is 0.9973.

    Returns:
        float: The Mahalanobis distance threshold.
    """
    return chi2.ppf(p, k)


def mahalanobis_distance(features: np.ndarray):
    """
    Computes the Mahalanobis distance between a feature and a mean vector.

    Parameters:
        features: array of the feature vector
        feature (np.ndarray): .
        mean (np.ndarray): The mean vector.
        cov (np.ndarray): The covariance matrix.

    Returns:
        float: The Mahalanobis distance.
    """

    mean = np.mean(features, axis=1)
    cov_layers = [LedoitWolf().fit(level) for level in features]

    diff = features - mean

    [mahalanobis()]

    return mahalanobis(diff, mean, np.linalg.inv(cov.covariance_))


if __name__ == "__main__":
    img_path = train_df["image_path"].iloc[0]
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    features = feature_extraction_model.predict(img)
    layers_shape = [layer.shape for layer in features]

    flattened_feats = np.concatenate([layer.flatten() for layer in features])
    rows, _ = train_df.shape

    with h5py.File(os.path.join(DATA_DIR, "mnt_ve3_tmp.h5"), "w") as h5:
        h5.create_dataset(
            "data", shape=(rows, flattened_feats.shape[0]), dtype=flattened_feats.dtype
        )

        for i in range(rows):
            img_path = train_df.iloc[i]["image_path"]
            img = image.load_img(img_path, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            predictions = feature_extraction_model.predict(img)
            features = np.concatenate([layer.flatten() for layer in features])

            flattened_feats = features.flatten()
            h5["data"][i, :] = flattened_feats
