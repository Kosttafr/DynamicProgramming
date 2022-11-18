import torch
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class PolicyNet(torch.nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(2, 24)
        self.act1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(24, 48)
        self.act2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(48, 11)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x

class QualityNet(torch.nn.Module):
    def __init__(self):
        super(QualityNet, self).__init__()
        self.fc1 = torch.nn.Linear(2, 24)
        self.act1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(24, 48)
        self.act2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(48, 11)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x