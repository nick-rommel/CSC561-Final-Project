import os
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
img_dir = 'C:/Users/alexa/Desktop/ELE_Masters/Spring_2023/CSC_561_Neural_Networks_and_Deep_Learning/MusicGenreClassification/ImageFolder/'
img_labels = os.listdir(img_dir)
img_labels = [[label,label[:label.index('0')]] for label in img_labels]
img_labels = [[img[0],0] if img[1] == 'blues'       else img for img in img_labels]
img_labels = [[img[0],1] if img[1] == 'classical'   else img for img in img_labels]
img_labels = [[img[0],2] if img[1] == 'country'     else img for img in img_labels]
img_labels = [[img[0],3] if img[1] == 'disco'       else img for img in img_labels]
img_labels = [[img[0],4] if img[1] == 'hiphop'      else img for img in img_labels]
img_labels = [[img[0],5] if img[1] == 'jazz'        else img for img in img_labels]
img_labels = [[img[0],6] if img[1] == 'metal'       else img for img in img_labels]
img_labels = [[img[0],7] if img[1] == 'pop'         else img for img in img_labels]
img_labels = [[img[0],8] if img[1] == 'reggae'      else img for img in img_labels]
img_labels = [[img[0],9] if img[1] == 'rock'        else img for img in img_labels]




class CustomImageDataset(Dataset):
    def __init__(self,img_labels,img_dir,transform = None,target_transform = None):
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        image = read_image(img_path)
        label = self.img_labels[idx][1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label,transforms = ToTensor)
        return image, label

CID = CustomImageDataset(img_labels,img_dir)

train_set,val_set,test_set = random_split(CID,[800,100,99])
train_loader = DataLoader(dataset = train_set,batch_size = 64,shuffle = True)
val_loader = DataLoader(dataset = val_set,batch_size = 64,shuffle = True)
test_loader = DataLoader(dataset = test_set,batch_size = 64,shuffle = True)
