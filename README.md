# AMLS_23-24_SN21185548

## Introduction 
Welcome to the project repository for the ELEC00134: Applied Machine Learning Systems coursework for the academic year 2023-2024. 
This project is focused on employing machine learning techniques, 
particularly transfer learning with ResNet-18 and ResNet-50, 
to classify medical images from the PneumoniaMNIST and PathMNIST datasets.

In Part A, ResNet-18 is used to classify gray-scale X-ray chest images for PneumoniaMNIST.

In Part B, ResNet-18 and ResNet-50 are used to classify PathMNIST dataset.

## Description
This repository contains all the necessary scripts and instructions to preprocess datasets, train machine learning models, and evaluate their performance. The aim is to achieve effective classification of gray-scale X-ray chest images for detecting pneumonia and various pathologies in histopathological images.

## Project Structure

Below is the file structure and descriptions for the project:

- Folder A contains python files for preprocessing PneumoniaMNIST, training model, and evaluating model.
- Folder B contains python files for preprocessing PathMNIST, training base model and specialised model, and evaluating model.
- Folder Datasets is kept empty to allow datasets to be pasted in for assessment.
- The root directory contains the `main.py` script that serves as the entry point for executing various functions within the project.

### A - PneumoniaMNIST
| File                            | Description                                     |
|---------------------------------|-------------------------------------------------|
| data_preprocessing_pneumonia.py | Script for preprocessing PneumoniaMNIST dataset |
| evaluation_pneumonia.py         | Script for evaluating the PneumoniaMNIST models |
| info.py                         | Information for PneumoniaMNIST dataset          |
| model_pneumonia.py              | PneumoniaMNIST model definitions                |
| train_pneumonia.py              | Training script for PneumoniaMNIST models       |

### B - PathMNIST
| File                       | Description                                               |
|----------------------------|-----------------------------------------------------------|
| data_preprocessing_path.py | Script for preprocessing PathMNIST dataset                |
| evaluation_path.py         | Script for evaluating the PathMNIST models                |
| info.py                    | Information for PathMNIST dataset                         |
| model_path.py              | PathMNIST model definitions                               |
| specialised_path.py        | Training script for class BACK, MUS, and STR of PathMNIST |
| train_path.py              | Training script for PathMNIST models                      |

## Getting Started

### Dependencies
The project was developed in Python 3.10.13. 
The project has been tested to work well with requirements.txt.
If you encounter difficulties when compiling the files, please see the specific versions of each library as follows:

```plaintext
torch==2.1.0 (with CUDA 12.3)
torchvision==0.16.0
matplotlib==3.8.0
tqdm==4.66.1
numpy==1.26.2
Pillow==10.0.1
seaborn==0.13.0
scikit-learn==1.3.0
```

### Installing
* To install dependencies, run:
```bash
pip install -r requirements.txt
```
* Paste pneumoniamnist.npz and pathmnist.npz files into Datasets folder
* Pre-trained models can be accessed at: https://drive.google.com/drive/folders/1tBu78HDPohnKhHxqT-NAbTC3Sqlcp2jX?usp=drive_link.

* Please download and paste them into folders as appropriate.

### Executing program

* Input prompts have been incorporated to help assess the performance of individual file within the project. 
Run `main.py` and follow the interactive prompts.
* Please choose the weights of model when trying to train models with/without pre-trained weights from PyTorch.
* (e.g. change self.model = models.resnet50(weights=**ResNet50_Weights.IMAGENET1K_V2**) to change self.model = models.resnet50(weights=**None**) )

