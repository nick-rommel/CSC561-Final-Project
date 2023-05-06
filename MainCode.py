# import statements
# importing the other files in this folder hierarchy
import CustomDataloader as CDL
from VITNet import VITNet as VN
import TrainEvalLoop as TEL
import time

# function to invoke training on the model
def train():
    # TODO make this path system agnostic. easiest way to do it is to probably have the data in the local folder
    #   This DOES require download the dataset locally and maintaining it that way.. i recommend configuring a .gitignore
    path = 'C:/Users/alexa/Desktop/ELE_Masters/Spring_2023/CSC_561_Neural_Networks_and_Deep_Learning/MusicGenreClassification/GTZAN/images_original/'
    
    # creating the dataloaders by invoking their function.
    # 64 images per patch, 800 train images, 100 validation, 99 test
    train_loader,val_loader,test_loader = CDL.CustomLoader(path,64,800,100,99)
    # instantiating the network.
    # TODO explain the hard-coded values
    Network = VN(1,4*256)

    # training the network using the different dataloaders.
    # learning rate = 0.05
    TEL.TrainNetwork(Network,0.05,train_loader,val_loader,test_loader)

# this file is the "main" function of the project, and is the entry point for execution.
if __name__ =='__main__':
    start = time.time()
    train()
    end = time.time()
    print(end-start)
