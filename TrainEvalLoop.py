import torch.optim as TO
import torch.nn as nn

def TrainNetwork(Network,LR,train_loader,val_loader,test_loader):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = TO.SGD(params = Network.parameters(),lr = LR)
    losses = []
    vlosses = []
    for epoch in range(5):
        Network.train()
        average_training_loss = Network.trainloop(train_loader,Network,criterion,optimizer)
        losses.extend(average_training_loss)
        Network.eval()
        average_validation_loss = Network.validationloop(val_loader,Network,criterion)
        vlosses.extend(average_validation_loss)
        print(average_validation_loss)
    print(vlosses)
