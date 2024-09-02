import os
import torch
from torch import nn
from torch.nn import Sequential,Conv2d,MaxPool2d,ReLU,Flatten,Linear,Softmax,CrossEntropyLoss,Dropout

class HandGestureCNN(nn.Module):
    def __init__(self,learning_rate=0.001):
        super(HandGestureCNN,self).__init__()
        self.model1=Sequential(
            Conv2d(1,32,5),
            ReLU(),
            MaxPool2d(2),
            Dropout(0.2),
            Conv2d(32,64,3),
            ReLU(),
            MaxPool2d(2),
            Dropout(0.2),
            Conv2d(64,64,3),
            ReLU(),
            MaxPool2d(2),
            Flatten(),
            Linear(64*10*10, 128), 
            ReLU(),
            Linear(128,10),
            Softmax(dim=1)
        )
        self.loss_fn=CrossEntropyLoss()
        self.optimizer=torch.optim.Adam(self.parameters(),lr=learning_rate)

    def forward(self, x):
        x = self.model1(x)
        return x
    
    def save_checkpoint(self, epoch, path='checkpoint.pth'):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load_checkpoint(self, path='checkpoint.pth'):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch'] + 1
        return 0
    
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.counter = 0
            
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
