import CustomDataloader as CDL
from MyNet import MyNet as MN
import TrainEvalLoop as TEL

path = 'C:/Users/alexa/Desktop/ELE_Masters/Spring_2023/CSC_561_Neural_Networks_and_Deep_Learning/MusicGenreClassification/GTZAN/images_original/'
train_loader,val_loader,test_loader = CDL.CustomLoader(path,100,800,100,99)
Network = MN(2,256)
TEL.TrainNetwork(Network,0.05,train_loader,val_loader,test_loader)
