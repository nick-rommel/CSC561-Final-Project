import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self,NumberOfLayers,Neurons):
        super(MyNet,self).__init__()
        self.input = nn.Sequential(nn.Flatten(start_dim=1),
                                   nn.Linear(373248,Neurons),
                                   nn.ReLU())
        self.hidden = nn.ModuleList([nn.Sequential(nn.Linear(Neurons,Neurons),
                                                   nn.ReLU()) for i in range(NumberOfLayers)])
        self.output = nn.Linear(Neurons,10)
    def forward(self,data):
        x = data.reshape(len(data),3,288,432)
        x = self.input(x)
        for layer in self.hidden:
            x = layer(x)
        guess = self.output(x)
        return guess
    def trainloop(self,dataset,model,criterion,optimizer):
        running_loss = []
        for data,labels in dataset:
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred,labels)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
        return running_loss
    def validationloop(self,dataset,model,criterion):
        running_loss = []
        for data,labels in dataset:
            pred = model(data)
            loss = criterion(pred,labels)
            running_loss.append(loss.item())
        return running_loss
    
