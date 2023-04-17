from torch.utils.data import DataLoader, random_split
from torchvision import datasets,transforms

path = 'C:/Users/alexa/Desktop/ELE_Masters/Spring_2023/CSC_561_Neural_Networks_and_Deep_Learning/MusicGenreClassification/GTZAN/images_original/'
transform = transforms.ToTensor()
dataset = datasets.ImageFolder(path,transform = transform)

train_set,val_set,test_set = random_split(dataset,[800,100,99])

train_loader = DataLoader(dataset = train_set,batch_size = 64,shuffle = True, num_workers = 4)
val_loader = DataLoader(dataset = val_set,batch_size = 64,shuffle = True, num_workers = 4)
test_loader = DataLoader(dataset = test_set,batch_size = 64,shuffle = True, num_workers = 4)
