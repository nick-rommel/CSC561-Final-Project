# import statements
# importing the other files in this folder hierarchy
import CustomDataloader as CDL
from VITNet import VITNet as VN
import time
import torch.optim as TO
import torch.nn as nn
import torch
import numpy as np
import wandb


# setting the device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# function to invoke training on the model
def train():
    # path to the dataset
    # this will need to be the absolute path to the dataset's "images_original" folder.
    path = 'C:/Masters/CSC561/Final Project/Code/CSC561-Final-Project/GTZAN/images_original/'

    # parameters that are remaining constant
    batch_size = 100
    num_epochs = 25    

    train_size = 800
    val_size = 100
    test_size = 99

    # creating the dataloaders by invoking their function.
    train_loader,val_loader,test_loader = CDL.CustomLoader(path,batch_size,train_size,val_size,test_size)

    # hyperparameters
    # we are looping over 46 different learning rates, and 11 different weight decays
    lr = np.arange(0.001, 0.0102, 0.0002)
    Weight_Decay = np.arange(0.005, 0.016, 0.001)
    best_param_dictionary = {
        "LR": lr[0],
        "weight_decay": Weight_Decay[0]
    }
    # setting up wandb
    # Followed guide on WandB website for all related content
    run = wandb.init(
        # setting the project to report the metrics to
        project="CSC561-Final-Project",

        # the tracked hyperparameters
        config={
            "Learning_Rate": lr,
            "Weight_Decay": Weight_Decay,
            "Epochs": num_epochs,
        })

    # declaring the variable to hold the best model
    # will be updating this each time a better model parameter combination is found in the loop below
    # Making use of the model.state_dict()
    best_model = VN(1,4*256)

    # variable for holding the current best validation accuracy
    best_val_acc = 0

    
    # hyperparameter loop
    for LR in lr:
        for weight_decay in Weight_Decay:

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
            optimizer = TO.AdamW(params=params,lr=LR, weight_decay=weight_decay)


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
                # ensuring that the model is set to "train" mode
                Network.train()

                # invoking the internal method defined for the training loop of the model
                average_training_loss,average_training_accuracy = trainloop(train_loader,Network,criterion,optimizer)
                # calculating actual average of the batch training accuracies and losses.
                # there are 8 minibatches per epoch.
                avg_train_loss = sum(average_training_loss) / 8
                avg_train_acc = sum(average_training_accuracy) / 8

                # appending returned values onto the end of their respective lists
                losses.extend(average_training_loss)
                accuracy.extend(average_training_accuracy)

                # changing the network to evaluation mode in order to calculate the validation accuracy
                Network.eval()
                with torch.no_grad():
                    average_validation_loss,average_validation_accuracy = validationloop(val_loader,Network,criterion)

                # if there is a new best val accuracy achieved, update current best accuracy and save the current
                #   model state
                # average_validation_accuracy is always a 1 element list.
                if average_validation_accuracy[0] > best_val_acc:
                    # Print statement to show that an update has been made
                    print(f"New best accuracy, updating metrics")
                    # setting the new best_val_acc
                    best_val_acc = average_validation_accuracy[0]
                    # saving the current weights and biases of the model that produced these new "best" results
                    best_model.load_state_dict(Network.state_dict())

                    # update the dictionary holding the best parameters
                    best_param_dictionary['LR']=LR
                    best_param_dictionary['weight_decay']=weight_decay

                # wandb logging
                # logging Validation and Test Accuracies/Losses for Each training epoch.
                wandb.log({"Validation_Accuracy": average_validation_accuracy[0],
                           "Validation_Loss": average_validation_loss[0],
                           "Training_Accuracy": avg_train_acc,
                           "Training_Loss": avg_train_loss})

                # appending the validation values to their respective lists.
                vlosses.extend(average_validation_loss)
                vaccuracy.extend(average_validation_accuracy)

        
        
            # defining reporting metrics
            name = f'LR:{LR},Weight_Decay:{weight_decay}'
            # calculating the running time of the training
            end = time.time()
            dur = end-start

            # printing these values
            print(name,f'Duration:{dur:0.2f},Acuracy:{accuracy[-1]:.0f},vAcuracy:{vaccuracy[-1]:.0f}',flush = True)

            # writing these metrics to a file for later use in graphing.
            filename = 'Train_Validation.txt'
            file = open(filename,'a',encoding='utf-8')
            file.write(f'{name}\t{losses}\t{vlosses}\t{accuracy}\t{vaccuracy}\t{dur}\n')
            file.close()

    # calling the evaluation of the test model using the best found model
    best_model.eval()
    with torch.no_grad():
        test_model(test_loader=test_loader,Network=best_model,criterion=criterion, dict=best_param_dictionary)



# Function for evaluating the best model stored
# This takes the snapshot of the best model recorded and loads it in as the current model state_dict
# Inputs:
#   test_loader: the test data's dataloader
#   Network: The "best" model (passed already in eval() mode)
#   dict: The dictionary holding the LR and weight-decay values used when the best model was found
# Outputs:
#   None
def test_model(test_loader, Network, criterion, dict):
    # declaring lists to hold the calculated metrics
    test_running_loss = []
    test_running_accuracy = []

    # sending the model to the GPU
    Network.to(device=device)

    # loop for going through the batches 
    for inputs,labels in test_loader:
        # for calculating the validation accuracy
        start = time.time()
        num_correct = 0
        num_samples = 0

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
        print(f'Final Inference on hold-out test set...')
        print(f'Test: {accuracy:0.0f}%, {dur:0.2f}seconds\n')

        name = f'LR:{dict["LR"]},Weight_Decay:{dict["weight_decay"]}'

        # writing these metrics to a file for later use.
        filename = 'Test.txt'
        file = open(filename,'a',encoding='utf-8')
        file.write(f'{name}\t{loss}\t{accuracy}\t{dur}\n')
        file.close()


# Function for executing the training of the model over the training batches.
# Inputs:
#   dataset: the dataloader
#   model: the model being used
#   criterion: the loss function (cross entropy)
# Outputs:
#   running_loss: the list of batch losses
#   running_accuracy: the list of batch accuracies
def trainloop(dataset,model,criterion,optimizer):
    # declaring lists for holding the intermediary metrics
    running_loss = []
    running_accuracy = []
    # going through the data batches in the dataloader
    for inputs,labels in dataset:
        # for the accuracy calculation
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

    # returning the lists of the calculated loss and accuracy
    return running_loss,running_accuracy



# function used for calculating the validation accuracies and loss
# Inputs:
#   dataset: the dataloader
#   model: the model being used
#   criterion: the loss function (cross entropy)
# Outputs:
#   running_loss: the list of batch losses (1 item in this case)
#   running_accuracy: the list of batch accuracies (1 item in this case)
def validationloop(dataset,model,criterion):
    # declaring lists to hold the calculated metrics
    running_loss = []
    running_accuracy = []

    # loop for going through the batches
    # for the validation loader, there is only a single batch due to the amount
    #   of available data.
    for inputs,labels in dataset:
        # for calculating the validation accuracy
        num_correct = 0
        num_samples = 0

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

    # returning the calculated loss and accuracy lists
    return running_loss,running_accuracy


    
# this file is the "main" function of the project, and is the entry point for execution.
if __name__ =='__main__':
    # logging into wandb
    wandb.login()

    start = time.time()
    train()
    end = time.time()
    print(end-start)
