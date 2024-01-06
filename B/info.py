# Adapted from MedMNIST github repository

__version__ = "2.2.3"

HOMEPAGE = "https://github.com/MedMNIST/MedMNIST/"

INFO = {
    "pathmnist": {
        "python_class": "PathMNIST",
        "description":
        "The PathMNIST is based on a prior study for predicting survival from colorectal cancer histology slides, providing a dataset (NCT-CRC-HE-100K) of 100,000 non-overlapping image patches from hematoxylin & eosin stained histological images, and a test dataset (CRC-VAL-HE-7K) of 7,180 image patches from a different clinical center. The dataset is comprised of 9 types of tissues, resulting in a multi-class classification task. We resize the source images of 3×224×224 into 3×28×28, and split NCT-CRC-HE-100K into training and validation set with a ratio of 9:1. The CRC-VAL-HE-7K is treated as the test set.",
        "url":
        "https://zenodo.org/record/6496656/files/pathmnist.npz?download=1",
        "MD5": "a8b06965200029087d5bd730944a56c1",
        "task": "multi-class",
        "label": {
            "0": "adipose",
            "1": "background",
            "2": "debris",
            "3": "lymphocytes",
            "4": "mucus",
            "5": "smooth muscle",
            "6": "normal colon mucosa",
            "7": "cancer-associated stroma",
            "8": "colorectal adenocarcinoma epithelium"
        },
        "n_channels": 3,
        "n_samples": {
            "train": 89996,
            "val": 10004,
            "test": 7180
        },
        "license": "CC BY 4.0"
    }
}
