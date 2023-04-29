import torch.optim as TO
import torch.nn as nn
import torch
import time

def TrainNetwork(Network,LR,train_loader,val_loader,test_loader):
    start = time.time()
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = TO.AdamW(params = Network.parameters(),lr = LR)
    losses = []
    vlosses = []
    accuracy = []
    vaccuracy = []
    for epoch in range(1):
        print(f'Epoch# {epoch+1}')
        Network.train()
        average_training_loss,average_training_accuracy = Network.trainloop(train_loader,Network,criterion,optimizer)
        losses.extend(average_training_loss)
        accuracy.extend(average_training_accuracy)
        Network.eval()
        with torch.no_grad():
            average_validation_loss,average_validation_accuracy = Network.validationloop(val_loader,Network,criterion)
        vlosses.extend(average_validation_loss)
        vaccuracy.extend(average_validation_accuracy)
    name = 'model_criteria'
    end = time.time()
    dur = end-start
    print(name,f'Duration:{dur:0.2f},Acuracy:{accuracy[-1]:.0f},vAcuracy:{vaccuracy[-1]:.0f}',flush = True)
    filename = 'TrainTest.txt'
    file = open(filename,'a',encoding='utf-8')
    file.write(f'{name}\t{losses}\t{vlosses}\t{accuracy}\t{vaccuracy}\t{dur}\n')
    file.close()
