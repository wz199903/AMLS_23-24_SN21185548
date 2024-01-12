# AMLS_23-24_SN21185548

Project repository for the 2023-2024 ELEC00134: Applied Machine Learning Systems coursework. 

In part A, ResNet-18 is used to classify gray-scale X-ray chest images for PneumoniaMNIST.

In part B, ResNet-18 and ResNet-50 are used to classify PathMNIST dataset.

Both tasks employ transfer learning to train the model.

## Description


## Project Structure

Below is the file structure and descriptions for the project:

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
| evaluation_pneumonia.py    | Script for evaluating the PathMNIST models                |
| info.py                    | Information for PathMNIST dataset                         |
| model_pneumonia.py         | Pneumonia model definitions                               |
| specialised_pneumonia.py   | Training script for class BACK, MUS, and STR of PathMNIST |
| train_pneumonia.py         | Training script for PathMNISTmodels                       |

## Getting Started

### Dependencies
The project was developed in Python 3.10.13 with the following packages. 
The project has been tested to work well with requirements.txt.
If you encounter difficulties when compiling the files, please see the specific versions of each library as follows:
* torch==2.1.0 (With CUDA 12.3)
* torchvision==0.16.0
* matplotlib==3.8.0
* tqdm==4.66.1
* numpy==1.26.2
* Pillow==10.0.1
* seaborn==0.13.0
* scikit-learn==1.3.0




### Installing
* To install dependencies, run:
```bash
pip install -r requirements.txt
```
* Paste pneumoniamnist.npz and pathmnist.npz files into Datasets folder
* Pre-trained models can be accessed at: https://drive.google.com/drive/folders/1tBu78HDPohnKhHxqT-NAbTC3Sqlcp2jX?usp=drive_link.
Please download and paste them into folders as appropriate.

### Executing program

* How to run the program
* Step-by-step bullets
