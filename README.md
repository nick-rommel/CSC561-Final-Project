# CSC561-Final-Project
Song Genre Classification using image classification on mel spectrograms.

This project employs transfer learning methodologies in order to use a pre-trained ViT and fine tune it on a much smaller, and also public, dataset.

## The Dataset

We used a modified version of the GTZAN dataset that already had its wav files converted into mel-spectrograms.

The dataset can be found at this [link](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).

The dataset consists of 1000 images, made up of 100 samples from 10 different genres of music.

## Getting Started with the code

In order to run our code, you will need to clone this repo and then follow these steps:

1. Download the dataset from the link in the section "The Dataset". You will need to write down the **absolute** path to where this dataset is stored.
2. On line 20 of `MainCode.py`, change the value assigned to `path` to be the absolute path to the GTZAN dataset you downloaded from the linked Kaggle page.
3. If you would like to run the version of our code with `WandB` reporting, you will need to make the above edits to the `MainCode-WandB.py` file and do the following:
    1. Our code integrates WandB for certain reporting metrics, so in order to run our code, you will need to have a API key for WandB as well as a project set up to store the results. After getting that information, change the `project` field on line 45 to be whatever name you desire.
    2. Finally, run the code and log into wandb when prompted by the terminal, and the code should begin running in earnest shortly after that.
4. If you do not wish to use `WandB` functionality, continue using the `MainCode.py` file.
5. **NOTE:** Currently, the hyperparameter sweep is set to train 506 different model combinations. It took ~8.5 hours to run on my RTX 3080ti with 4 workers enabled for each dataloader. Feel free to change the range of the hyperparameter sweep in lines 35-36 if you do not desire to run the code for as long as we did.

## Getting to know the Repo:

### `MainCode-WandB.py`:

This file is the `main` file of our project, and serves as the entry point for the Runtime Execution. 

This file contains the code for running the hyperparameter sweep to train the different model combinations, and then uses the "best found" model weights and biases for inference on the hold-out Test data split.

Additionally, this version of this `main` file contains `WandB` setup and reporting.

### `MainCode.py`:

This file is nearly identical to `MainCode-WandB.py`. The difference is that it does not contain `WandB` functionality. Use this file if you just want to run the hyperparameter sweep *without* reporting to the `WandB` website.

### `CustomDataloader.py`:

This module is what we use to load the data into Dataloaders for Training, Validation, and a hold-out Test split.

### `VITNey.py`:

This file is where we handle the downloading/transfer learning of the ViT_b16 from Google's original ViT experiment. We lop off the prediction head of the original model, but keep the pretrained weights and biases for the main transformer architecture. We replace the prediction head with our own linear layer combination into 10 output classes.

### `run_data` Folder:

This folder contains 4 files:

- `Test_Full_Spread.txt` which contains the final accuracy and loss of the best model's inference on the test data, as well as the Learning Rate and Weight Decay values that resulted in the best model. This document contains the testing metrics for the Original, Full Hyperparamter sweep.
- `Train_Eval_Full_Spread.txt` which contains all of the recorded run data from training the models during the hyperparameter sweep. The metrics recorded for each model were: The Learning Rate and Weight Decay combination, the Training/Validation Accuracies and Losses, and the length of time it took to train the model. This file contains the Training and Validation metrics for the Original, Full Hyperparameter sweep.
- `Test.txt` which contains the testing metrics for the second, focused hyperparameter sweep that was using validation loss, instead of validation accuracy, as the evaluation metric for choosing the best model.
- `Train_Validation.txt` which contains the Training and Validation metrics for the the second, focused hyperparameter sweep that was using validation loss, instead of validation accuracy, as the evaluation metric for choosing the best model.

### `Images` Folder:

This folder contains all of the graphs we generated from the data found in the `run_data` folder. 

The `Sweep 1` folder contains all of the images generated from the original, large Hyperparameter sweep. This folder also contains subfolders for the organization of the images.

The `Sweep 2` folder contains all of the images generated from the second sweep that used validation loss, instead of validation accuracy, as the evaluation metric for choosing the best model. This folder also contains subfolders for the organization of the images.

The `Overview of Image Folder.docx` contains a description of the images in all of the subfolders.

### The Different Branches:

The `main` branch is as expected, the most current and up-to-date version of our code (currently baselined off of the `wandb-dev` branch).

The `wandb-dev` branch is the current development branch in which the final code changes to incorporate WandB were made.

The `development`, `nick-dev`, and `alex-dev` branches have been deprecated.