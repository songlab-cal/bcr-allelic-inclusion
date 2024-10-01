import torch                                                                                                      
from utils import ALPHABET

class CNN(torch.nn.Module):
    def __init__(self, use_vj=True, out_dim=3, num_v=None, num_j=None):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(len(ALPHABET), 60, 5, padding=2)
        self.conv2 = torch.nn.Conv1d(60, 80, 3, padding=1)
        self.conv3 = torch.nn.Conv1d(80, 80, 3, padding=1)
        self.conv4 = torch.nn.Conv1d(80, 100, 3, padding=1)
        if use_vj:
            self.fc1 = torch.nn.Linear(4*100 + num_v + num_j, 20) 
        else:
            self.fc1 = torch.nn.Linear(4*100, 20) 
        self.fc2 = torch.nn.Linear(20, out_dim)
        self.use_vj = use_vj
        self.dropout = torch.nn.Dropout(0.1)

    def conv_embed(self, x): 
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool1d(x, 2)

        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool1d(x, 2)

        x = self.conv3(x)
        x = torch.nn.functional.relu(x)

        x = self.conv4(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool1d(x, 2)

        x = torch.flatten(x, 1)

        return x

    def embed(self, x, v, j):
        x = self.conv_embed(x)
        if self.use_vj:
            x = torch.cat([x, v, j], dim=1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        
        return x

    def forward(self, x, v, j):
        x = self.conv_embed(x)
        if self.use_vj:
            x = torch.cat([x, v, j], dim=1)

        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

