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
    transform = transforms.Compose([
        transforms.ToTensor()
        ])
    
    # executing the data transform
    dataset = datasets.ImageFolder(path,transform = transform)

    # utilizing random_split to split the data into the three groups.
    # TODO discuss if we should do the hold-out test set on our own
    train_set,val_set,test_set = random_split(dataset,[train_size,val_size,test_size])
    train_loader =  DataLoader(dataset = train_set, batch_size = batch_size,    shuffle = True, num_workers = 4)
    val_loader =    DataLoader(dataset = val_set,   batch_size = val_size,      shuffle = True, num_workers = 4)
    test_loader =   DataLoader(dataset = test_set,  batch_size = test_size,     shuffle = True, num_workers = 4)

    # returning the dataloaders
    return train_loader,val_loader,test_loader
