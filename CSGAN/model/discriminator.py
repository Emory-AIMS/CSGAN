import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    def __init__(self,
                 num_clusters,
                 total_locations,
                 embedding_dim,
                 hidden_dim,
                 num_layers,
                 bidirectional,
                 traj_len,
                 dropout,
                 device):
        super(Discriminator, self).__init__()
        self.num_clusters = num_clusters
        self.total_locations = total_locations
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.traj_len = traj_len
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(total_locations, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional,
                          dropout=dropout)
        self.linear_hidden_dim = 2*num_layers*hidden_dim if bidirectional else num_layers*hidden_dim
        self.linear_1 = nn.Linear(self.linear_hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.linear_2 = nn.Linear(hidden_dim, self.num_clusters+1)

    def init_hidden(self, batch_size):
        """
        Initializes hidden state for GRU.
        """
        h = autograd.Variable(torch.zeros(2*self.num_layers if self.bidirectional
                                          else self.num_layers, batch_size, self.hidden_dim)).to(self.device)
        return h

    def forward(self, x, h):
        """
        Forward pass of the discriminator.
        :param x: batch_size * seq_len
        """
        # batch_size x seq_len x embedding_dim
        embedded_input = self.embedding(x)
        # seq_len * batch_size * embedding_dim
        embedded_input = embedded_input.permute(1, 0, 2)
        # h: (2*num_layers if bidirectional else num_layers) * batch_size * hidden_dim
        _, h = self.gru(embedded_input, h)
        # batch_size * (2*num_layers if bidirectional else num_layers) * hidden_dim
        h = h.permute(1, 0, 2).contiguous()
        pred = self.linear_1(h.view(-1, self.linear_hidden_dim))
        pred = torch.tanh(pred)
        pred = self.dropout(pred)
        # batch_size * (num_clusters+1)
        pred = self.linear_2(pred)
        m = nn.LogSoftmax(dim=1)
        pred = m(pred)
        return pred

    def batchClassify(self, x):
        """
        Classifies a batch of sequences.
        Inputs:
            - x: batch_size * seq_len
        Returns: out
            - pred: batch_size (class label -> real cluster id (from 1 to num_cluster)
              for real trajectory and fake cluster id {num_cluster+1} for generated trajectory)
        """
        batch_size = x.size()[0]
        h = self.init_hidden(batch_size)
        pred = self.forward(x, h)
        return pred

    def batchNLLLoss(self, x, label):
        """
        Returns NLL Loss for discriminator.
         Inputs:
            - x: batch_size * seq_len
            - label: batch_size (class label -> real cluster id (from 1 to num_cluster)
              for real trajectory and fake cluster id {num_cluster+1} for generated trajectory)
        """
        criterion = nn.NLLLoss()
        # criterion = nn.CrossEntropyLoss()
        batch_size = x.size()[0]
        h = self.init_hidden(batch_size)
        pred = self.forward(x, h)
        return criterion(pred, label)
    
    def rewardGeneration(self, x):
        batch_size = x.size()[0]
        h = self.init_hidden(batch_size)
        pred = self.reward_forward(x, h)
        return pred
    
    def reward_forward(self, x, h):
        # batch_size x seq_len x embedding_dim
        embedded_input = self.embedding(x)
        # seq_len * batch_size * embedding_dim
        embedded_input = embedded_input.permute(1, 0, 2)
        # h: (2*num_layers if bidirectional else num_layers) * batch_size * hidden_dim
        _, h = self.gru(embedded_input, h)
        # batch_size * (2*num_layers if bidirectional else num_layers) * hidden_dim
        h = h.permute(1, 0, 2).contiguous()
        pred = self.linear_1(h.view(-1, self.linear_hidden_dim))
        pred = torch.tanh(pred)
        pred = self.dropout(pred)
        # batch_size * (num_clusters+1)
        pred = self.linear_2(pred)
        m = nn.Softmax(dim=1)
        pred = m(pred)
        return pred



