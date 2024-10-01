import torch                                                                                                      
from utils import ALPHABET

class CNN_Paired(torch.nn.Module):
    def __init__(self, use_vj=True, out_dim=2, num_v=None, num_j=None):
        super(CNN_Paired, self).__init__()
        self.conv1 = torch.nn.Conv2d(len(ALPHABET), 20, (2,5), padding=(1,2))
        self.conv2 = torch.nn.Conv2d(20, 20, (2,3), padding=(0,1))
        if use_vj:
            self.fc1 = torch.nn.Linear(4*20 + num_v + num_j, out_dim)
        else:
            self.fc1 = torch.nn.Linear(4*20, out_dim) 
        self.use_vj = use_vj
        self.dropout = torch.nn.Dropout(0.3)

    def conv_embed(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, (1, 4))

        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, (2,2))

        x = torch.flatten(x, 1)

        return x

    def forward(self, x, v, j):
        x = self.conv_embed(x)
        if self.use_vj:
            x = torch.cat([x, v, j], dim=1)

        x = self.fc1(x)
        return x

