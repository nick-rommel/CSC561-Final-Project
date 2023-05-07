# import statements
# importing the other files in this folder hierarchy
import CustomDataloader as CDL
from VITNet import VITNet as VN
import time
import torch.optim as TO
import torch.nn as nn
import torch
import os
import numpy as np
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial


# setting the device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# function to invoke training on the model
def train():
    # path to the dataset
    path = 'C:/Masters/CSC561/Final Project/Code/CSC561-Final-Project/GTZAN/images_original/'

    # parameters
    batch_size = 100
    num_epochs = 10    

    train_size = 800
    val_size = 100
    test_size = 99

    # creating the dataloaders by invoking their function.
    train_loader,val_loader,test_loader = CDL.CustomLoader(path,batch_size,train_size,val_size,test_size)

    # hyperparameters
    lr = 0.001
    
    # instantiating the network.
    Network = VN(1,4*256)

    # variable for invoking loss calculations
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # in order to only train the 'head', hidden, and output layers, we define the parameters for the optimizer as such
    # this means the ViT isn't updated as the model learns.
    params = [
        {'params':Network.vit.heads.parameters()},
        {'params':Network.hidden.parameters()},
        {'params':Network.output.parameters()}
        ]
    
    # Using AdamW as the optimizer of choice. AdamW is often used as the optimizer in
    #   applications making use of the ViT, as it was used in ViT's introductory paper
    optimizer = TO.AdamW(params=params,lr=lr)


    ################# MODEL TRAINING ################
    # variable for timing metrics
    start = time.time()

    # delcaring lists to hold the loss values for use in the future.
    losses = []
    vlosses = []
    accuracy = []
    vaccuracy = []

    # sending the model to the GPU
    Network.to(device=device)

    # Below is the actual training loop for the model.
    for epoch in range(num_epochs):
        # Printing which epoch we are on
        print(f'Epoch# {epoch+1}')

        # ensuring that the model is set to "train" mode
        Network.train()

        # invoking the internal method defined for the training loop of the model
        average_training_loss,average_training_accuracy = trainloop(train_loader,Network,criterion,optimizer)

        # appending returned values onto the end of their respective lists
        losses.extend(average_training_loss)
        accuracy.extend(average_training_accuracy)

    # changing the network to evaluation mode in order to calculate the validation accuracy
    Network.eval()
    with torch.no_grad():
        average_validation_loss,average_validation_accuracy = validationloop(val_loader,Network,criterion)

    # appending the validation values to their respective lists.
    vlosses.extend(average_validation_loss)
    vaccuracy.extend(average_validation_accuracy)
    
    # defining reporting metrics
    name = 'model_criteria'

    # calculating the running time of the training
    end = time.time()
    dur = end-start

    # printing these values
    print(name,f'Duration:{dur:0.2f},Acuracy:{accuracy[-1]:.0f},vAcuracy:{vaccuracy[-1]:.0f}',flush = True)

    # writing these metrics to a file for later use.
    filename = 'TrainTest.txt'
    file = open(filename,'a',encoding='utf-8')
    file.write(f'{name}\t{losses}\t{vlosses}\t{accuracy}\t{vaccuracy}\t{dur}\n')
    file.close()

    #################### TESTING THE NETWORK WITH FINAL CONFIGURATION ##########################
    # declaring lists to hold the calculated metrics
    test_running_loss = []
    test_running_accuracy = []
    minibatch = 0

    # sending the model to the GPU
    Network.to(device=device)

    # loop for going through the batches 
    for inputs,labels in test_loader:
        # for calculating the validation accuracy
        start = time.time()
        num_correct = 0
        num_samples = 0
        minibatch += 1

        # sending the data to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # exectuing the prediction and calculating the loss.
        pred = Network(inputs)
        loss = criterion(pred,labels)
        test_running_loss.append(loss.item())

        # calculating the accuracy and recording it
        _,predictions = pred.cpu().detach().max(1)
        num_correct += (predictions == labels.cpu().detach()).sum()
        num_samples += predictions.size(0)
        accuracy = float(num_correct/num_samples)*100
        test_running_accuracy.append(accuracy)
        end = time.time()
        dur = end - start
        print(f'Test: {accuracy:0.0f}%, {dur:0.2f}seconds\n')
    
def trainloop(dataset,model,criterion,optimizer):
    # declaring lists for holding the intermediary metrics
    running_loss = []
    running_accuracy = []
    minibatch = 0
    # going through the data batches in the dataloader
    for inputs,labels in dataset:
        # for the accuracy calculation
        start = time.time()
        minibatch += 1
        num_correct = 0
        num_samples = 0


        # initializing the gradients to 0.
        optimizer.zero_grad()

        # sending the data to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # executing the prediction and and calculating the loss
        pred = model(inputs)
        loss = criterion(pred,labels)
        loss.backward()
        optimizer.step()

        # calculating the accuracy
        _,predictions = pred.cpu().detach().max(1)
        num_correct += (predictions == labels.cpu().detach()).sum()
        num_samples += predictions.size(0)
        accuracy = float(num_correct/num_samples)*100

        # appending the training accuracy and loss
        running_accuracy.append(accuracy)
        running_loss.append(loss.item())
        end = time.time()
        dur = end - start
        print(f' Training Batch {minibatch}: {accuracy:0.0f}%, {dur:0.0f}seconds')

    # returning the lists of the calculated loss and accuracy
    return running_loss,running_accuracy
    
# function used for calculating the validation accuracies and loss
# takes as input:
#   dataset: the dataloader
#   model: the model being used
#   criterion: the loss function (cross entropy)
def validationloop(dataset,model,criterion):
    # declaring lists to hold the calculated metrics
    running_loss = []
    running_accuracy = []
    minibatch = 0

    # loop for going through the batches 
    for inputs,labels in dataset:
        # for calculating the validation accuracy
        start = time.time()
        num_correct = 0
        num_samples = 0
        minibatch += 1

        # sending the data to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # exectuing the prediction and calculating the loss.
        pred = model(inputs)
        loss = criterion(pred,labels)
        running_loss.append(loss.item())

        # calculating the accuracy and recording it
        _,predictions = pred.cpu().detach().max(1)
        num_correct += (predictions == labels.cpu().detach()).sum()
        num_samples += predictions.size(0)
        accuracy = float(num_correct/num_samples)*100
        running_accuracy.append(accuracy)
        end = time.time()
        dur = end - start
        print(f'Validation: {accuracy:0.0f}%, {dur:0.2f}seconds\n')

    # returning the calculated loss and accuracy lists
    return running_loss,running_accuracy
    
# this file is the "main" function of the project, and is the entry point for execution.
if __name__ =='__main__':
    start = time.time()
    train()
    end = time.time()
    print(end-start)
