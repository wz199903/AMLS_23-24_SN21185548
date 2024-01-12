# Adapted from MedMNIST GitHub repository


__version__ = "2.2.3"

HOMEPAGE = "https://github.com/MedMNIST/MedMNIST/"

INFO = {
    "pneumoniamnist": {
        "python_class": "PneumoniaMNIST",
        "description":
        "The PneumoniaMNIST is based on a prior dataset of 5,856 pediatric chest X-Ray images. The task is binary-class classification of pneumonia against normal. We split the source training set with a ratio of 9:1 into training and validation set and use its source validation set as the test set. The source images are gray-scale, and their sizes are (384−2,916)×(127−2,713). We center-crop the images and resize them into 1×28×28.",
        "url":
        "https://zenodo.org/record/6496656/files/pneumoniamnist.npz?download=1",
        "MD5": "28209eda62fecd6e6a2d98b1501bb15f",
        "task": "binary-class",
        "label": {
            "0": "normal",
            "1": "pneumonia"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 4708,
            "val": 524,
            "test": 624
        },
        "license": "CC BY 4.0"
    },
}
