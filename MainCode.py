# import statements
# importing the other files in this folder hierarchy
import CustomDataloader as CDL
from VITNet import VITNet as VN
import TrainEvalLoop as TEL
import time
import torch.optim as TO
import torch.nn as nn

# function to invoke training on the model
def train():
    # path to the dataset
    path = 'C:/Masters/CSC561/Final Project/Code/CSC561-Final-Project/GTZAN/images_original/'
    
    # creating the dataloaders by invoking their function.

    # hyperparamters
    batch_size = 100
    lr = 0.001
    num_epochs = 20

    train_size = 800
    val_size = 100
    test_size = 99

    train_loader,val_loader,test_loader = CDL.CustomLoader(path,batch_size,train_size,val_size,test_size)
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

    # training the network using the different dataloaders.
    # passing all of the other hyperparameters that weren't the batch size
    TEL.TrainNetwork(Network=Network,LR=lr,train_loader=train_loader,val_loader=val_loader,test_loader=test_loader, \
                     epochs=num_epochs,criterion=criterion,optimizer=optimizer)

    # after the best hyperparameters have been found
# this file is the "main" function of the project, and is the entry point for execution.
if __name__ =='__main__':
    start = time.time()
    train()
    end = time.time()
    print(end-start)
