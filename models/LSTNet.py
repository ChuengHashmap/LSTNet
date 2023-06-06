import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.gru_layers = args.GRU_layers
        self.P = args.window
        self.m = data.m
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.Ck = args.CNN_kernel
        # in_channel, out_channel, kernel_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR, batch_first=True, num_layers=self.gru_layers)
        self.dropout = nn.Dropout(p=args.dropout)
        self.linear1 = nn.Linear(self.hidR, self.m)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh

    def forward(self, x):
        x = torch.reshape(x, (-1, 1, 168, 8))
        x = self.conv1(x)
        x = F.relu(x)
        x = x.squeeze(3)
        x = x.permute(0, 2, 1)
        if self.gru_layers != 1:
            x = self.GRU1(x)[-1][1]
        else:
            x = self.GRU1(x)[1]
        # print("GRU1(x)", x.size())
        x = self.dropout(x)
        x = self.linear1(x)
        # print("linear1(x)", x.size())
        res = x
        return res
