# import statements
import torch.optim as TO
import torch.nn as nn
import torch
import time

# This function acts as the main training loop for the model
# It takes as input:
#   Network: the model being used
#   LR: The learning rate
#   train_loader: the dataloader for the training split of the data
#   val_loader: the dataloader for the validation split of the data
#   test_loader: the dataloader for the hold-out test split of the data
def TrainNetwork(Network,LR,train_loader,val_loader,test_loader):
    # variable for timing metrics
    start = time.time()

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
    optimizer = TO.AdamW(params = params,lr = LR)

    # delcaring lists to hold the loss values for use in the future.
    losses = []
    vlosses = []
    accuracy = []
    vaccuracy = []

    # declaring variable for number of epochs.
    num_epochs = 1

    # Below is the actual training loop for the model.
    for epoch in range(num_epochs):
        # Printing which epoch we are on
        print(f'Epoch# {epoch+1}')

        # ensuring that the model is set to "train" mode
        Network.train()

        # invoking the internal method defined for the training loop of the model
        average_training_loss,average_training_accuracy = Network.trainloop(train_loader,Network,criterion,optimizer)

        # appending returned values onto the end of their respective lists
        losses.extend(average_training_loss)
        accuracy.extend(average_training_accuracy)

        # changing the network to evaluation mode in order to calculate the validation accuracy
        Network.eval()
        with torch.no_grad():
            average_validation_loss,average_validation_accuracy = Network.validationloop(val_loader,Network,criterion)

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
