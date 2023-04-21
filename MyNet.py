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
        running_accuracy = []
        for i,data in enumerate(dataset):
            num_correct = 0
            num_samples = 0
            inputs,labels = data
            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred,labels)
            loss.backward()
            optimizer.step()
            _,predictions = pred.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)
            accuracy = float(num_correct/num_samples)*100
            running_accuracy.append(accuracy)
            running_loss.append(loss.item())
            print(f' Training Batch {i+1}: {accuracy}')
        return running_loss
    def validationloop(self,dataset,model,criterion):
        running_loss = []
        running_accuracy = []
        for i,data in enumerate(dataset):
            num_correct = 0
            num_samples = 0
            inputs,labels = data
            pred = model(inputs)
            loss = criterion(pred,labels)
            running_loss.append(loss.item())
            _,predictions = pred.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)
            accuracy = float(num_correct/num_samples)*100
            running_accuracy.append(accuracy)
            print(f'Validation: {accuracy}\n')
        return running_loss
    

