from torch.utils.data import DataLoader, random_split
from torchvision import datasets,transforms

def CustomLoader(path,batch_size,train_size,val_size,test_size):
    transform = transforms.Compose([
        transforms.ToTensor()
        ])
    dataset = datasets.ImageFolder(path,transform = transform)
    train_set,val_set,test_set = random_split(dataset,[train_size,val_size,test_size])
    train_loader =  DataLoader(dataset = train_set, batch_size = batch_size,    shuffle = True, num_workers = 4)
    val_loader =    DataLoader(dataset = val_set,   batch_size = val_size,      shuffle = True, num_workers = 4)
    test_loader =   DataLoader(dataset = test_set,  batch_size = test_size,     shuffle = True, num_workers = 4)
    return train_loader,val_loader,test_loader
