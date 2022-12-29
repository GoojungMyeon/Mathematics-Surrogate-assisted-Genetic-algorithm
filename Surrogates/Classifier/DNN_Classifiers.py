import torch


class DNN_Classifiers(torch.nn.Module):
    def __init__(self, input_dim):
        super(DNN_Classifiers, self).__init__()
        self.l1 = torch.nn.Linear(input_dim, 64)
        self.l2 = torch.nn.Linear(64, 64)
        self.l3 = torch.nn.Linear(64, 32)
        self.l4 = torch.nn.Linear(32, 16)
        self.l5 = torch.nn.Linear(16, 8)
        self.l6 = torch.nn.Linear(8, 1)

    def forward(self, x):
        x = (self.l1(x))
        x = torch.nn.functional.relu(self.l2(x))
        x = torch.nn.functional.relu(self.l3(x))
        x = torch.nn.functional.relu(self.l4(x))
        x = torch.nn.functional.relu(self.l5(x))
        x = (self.l6(x))
        return x
