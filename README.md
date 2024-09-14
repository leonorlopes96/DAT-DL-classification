
# DAT-DL-classification
by Leonor Lopes

This project aims to classify Dopamine Transporter (DAT) brain scans 
using a convolutional neural network, specifically a ResNet architecture. 

The model can be used as an AI-assisted diagnostic tool to identify neurological conditions such as Parkinson's disease, Multiple System Atrophy or Progressive Supranuclear Palsy.


## Installation

Follow these steps to set up the project on your local machine:

1. Clone the repository:
   ```
   git clone https://github.com/leonorlopes96/DAT-DL-classification.git
   cd DAT-DL-classification
   ```


2. Install the required packages in a conda environment using the `requirements.yml` file:
   ```
   conda env update --file requirements.yml
   ```

   If you're not using conda, you can use pip:
   ```
   pip install -r requirements.txt
   ```


## Dataset Preparation

1. Create a csv file with columns 'img_paths' - where the filepaths will be located - and 'labels' with the classes 
of each corresponding file.
2. Create a .ini config file with training parameters and directories to save the results. An example is shown below:
```
[VAR]

BATCH_SIZE = 4
INIT_LR = 0.0001
EPOCHS = 1
USE_WEIGHTS = True
EARLY_STOP_PATIENCE = 30
LR_DROP = 0.5
LR_PATIENCE = 10

[DIR]

PROJ_DIR = /path/to/directory/to/save/results
TRAIN_DIR = /path/to/directory/to/save/train/results
TEST_DIR = /path/to/directory/to/save/test/results
TRAIN_DATA_DIR = /path/to/csv/file/with/train/filepaths.csv
TEST_DATA_DIR = /path/to/csv/file/with/test/filepaths.csv

TRAIN_DATA_MODE = 'gen'
```

## Train/Test the model

1. Add the path to your config file in main.py
2. Run main.py





