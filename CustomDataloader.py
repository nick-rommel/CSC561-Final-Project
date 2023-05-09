# import statements
from torch.utils.data import DataLoader, random_split
from torchvision import datasets,transforms


# function for setting up the different dataloaders for the train, val, and test data splits
# Inputs:
#   batch_size: the batch size
#   train_size: the size of the training split
#   val_size: the size of the validation split
#   test_size: the size of the test split
# Returns:
#  the three dataloaders.
def CustomLoader(path,batch_size,train_size,val_size,test_size):
    # transforming the input data 
    # we need to resize the data to 224x224 to fit into the ViT
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop([218,348]),# we get 218x348 image centered on the original image. This get rid of most of the 'white' border, but still leaves 10pixels width on the left side of the image
        transforms.Resize([224,224],antialias=True),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) #this normilization was taken from PyTorch Vit_B_16 transforms information
        ])
    
    # executing the data transform
    dataset = datasets.ImageFolder(path,transform = transform)

    # utilizing random_split to split the data into the three groups.
    train_set,val_set,test_set = random_split(dataset,[train_size,val_size,test_size])
    train_loader =  DataLoader(dataset = train_set, batch_size = batch_size,    shuffle = True, num_workers = 4)
    val_loader =    DataLoader(dataset = val_set,   batch_size = val_size,      shuffle = True, num_workers = 4)
    test_loader =   DataLoader(dataset = test_set,  batch_size = test_size,     shuffle = True, num_workers = 4)

    # we open the loader up here, so we don't have to do it in the actual training loops
    # this sped up the loop significantly.
    train_data = [data for data in train_loader]
    val_data = [data for data in val_loader]
    test_data = [data for data in test_loader]

    # returning the dataloaders
    return train_data,val_data,test_data

