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
3. Our code integrates WandB for certain reporting metrics, so in order to run our code, you will need to have a API key for WandB as well as a project set up to store the results. After getting that information, change the `project` field on line 45 to be whatever name you desire.
4. Finally, run the code and log into wandb when prompted by the terminal, and the code should begind running in earnest shortly after that.
5. **NOTE:** Currently, the hyperparameter sweep is set to train 506 different model combinations. It took ~8.5 hours to run on my RTX 3080ti with 4 workers enabled for each dataloader. Feel free to change the range of the hyperparameter sweep in lines 35-36 if you do not desire to run the code for as long as we did.

## Getting to know the Repo:

### `MainCode.py`:

This file is the `main` file of our project, and serves as the entry point for the Runtime Execution. 

This file contains the code for running the hyperparameter sweep to train the different model combinations, and then uses the "best found" model weights and biases for inference on the hold-out Test data split.

### `CustomDataloader.py`:

This module is what we use to load the data into Dataloaders for Training, Validation, and a hold-out Test split.

### `VITNey.py`:

This file is where we handle the downloading/transfer learning of the ViT_b16 from Google's original ViT experiment. We lop off the prediction head of the original model, but keep the pretrained weights and biases for the main transformer architecture. We replace the prediction head with our own linear layer combination into 10 output classes.

### `run_data` Folder:

This folder contains two files:

- `Test.txt` which contains the final accuracy and loss of the best model's inference on the test data, as well as the Learning Rate and Weight Decay values that resulted in the best model.
- `Train_Validation.txt` which contains all of the recorded run data from training the models during the hyperparameter sweep. The metrics recorded for each model were: The Learning Rate and Weight Decay combination, the Training/Validation Accuracies and Losses, and the length of time it took to train the model.

### `images` Folder:

This folder contains all of the graphs we generated from the data found in the `run_data` folder.

### The Different Branches:

The `main` branch is as expected, the most current and up-to-date version of our code (currently baselined off of the `wandb-dev` branch).

The `wandb-dev` branch is the current development branch in which the final code changes to incorporate WandB were made.

The `development`, `nick-dev`, and `alex-dev` branches have been deprecated.