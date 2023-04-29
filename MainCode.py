import CustomDataloader as CDL
from MyNet import MyNet as MN
import TrainEvalLoop as TEL
import time

def train():
    path = 'C:/Users/alexa/Desktop/ELE_Masters/Spring_2023/CSC_561_Neural_Networks_and_Deep_Learning/MusicGenreClassification/GTZAN/images_original/'
    train_loader,val_loader,test_loader = CDL.CustomLoader(path,64,800,100,99)
    Network = MN(2,4*256)
    TEL.TrainNetwork(Network,0.05,train_loader,val_loader,test_loader)

if __name__ =='__main__':
    start = time.time()
    train()
    end = time.time()
    print(end-start)
