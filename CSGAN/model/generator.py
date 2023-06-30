import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class Generator(nn.Module):
    def __init__(self,
                 origin,
                 bidirectional,
                 embedding_dim,
                 hidden_dim,
                 total_locations,
                 traj_len,
                 device):
        super(Generator, self).__init__()
        self.origin = origin
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.traj_len = traj_len
        self.total_locations = total_locations
        self.device = device

        self.embeddings = nn.Embedding(total_locations, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, bidirectional=bidirectional)
        self.linear = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, total_locations)

    def forward(self, x, h):
        """
        Generator the next possible visit given the current visit
        """
        # 1 * batch_size * embedding_dim
        embedded_input = self.embeddings(x).unsqueeze(0)
        # 1 * batch_size * (hidden_dim * 2 if bidirectional else hidden_dim)
        pred, h = self.gru(embedded_input, h)
        pred = self.linear(pred.view(-1, 2 * self.hidden_dim if self.bidirectional else self.hidden_dim))
        # m = nn.Softmax(dim=1)
        # pred = m(pred)
        pred = F.log_softmax(pred, dim=1)
        return pred, h

    def init_hidden(self, batch_size):
        """
        Initialize hidden state
        """
        h = autograd.Variable(torch.zeros(2 if self.bidirectional else 1, batch_size, self.hidden_dim)).to(self.device)
        return h

    def sample(self, num_samples):
        """
        Generate a group of trajectories using the generator
        """
        h = self.init_hidden(num_samples)
        gen_samples = torch.zeros(num_samples, self.traj_len).long().to(self.device)
        if self.origin == 'zero':
            x = autograd.Variable(torch.LongTensor([1] * num_samples)).to(self.device)
        elif self.origin == 'random':
            x = autograd.Variable(torch.LongTensor(np.random.randint(1, self.total_locations, num_samples))).to\
                (self.device)
        else:
            raise ValueError('currently we only support zero and random origin')
        for i in range(self.traj_len):
            # num_samples * total_locations
            pred, h = self.forward(x, h)
            pred = torch.exp(pred)
            # num_samples * 1
            pred = torch.multinomial(pred, 1)
            # pred = torch.argmax(pred, dim=1)
            gen_samples[:, i] = pred.view(-1).data
            x = pred.view(-1)
        return gen_samples

    def batchNLLLoss(self, gen_input, gen_target):
        """
        Returns the NLL Loss for predicting target sequence.
        Inputs:
            - gen_input: batch_size * seq_len
            - gen_target: batch_size * seq_len
            gen_input should be the same as gen_target with origin prepended
        """
        criterion = nn.NLLLoss()
        # criterion = nn.CrossEntropyLoss()
        batch_size, seq_len = gen_input.size()
        # seq_len * batch_size
        gen_input = gen_input.permute(1, 0)
        gen_target = gen_target.permute(1, 0)
        h = self.init_hidden(batch_size)
        loss = 0.0
        for i in range(seq_len):
            pred, h = self.forward(gen_input[i], h)
            loss += criterion(pred, gen_target[i])
        return loss

    def batchPGLoss(self, gen_input, gen_target, reward):
        """
        Returns a pseudo-loss that gives corresponding policy gradients.
        Inputs:
            - gen_input: batch_size * seq_len
            - gen_target: batch_size * seq_len
            - reward: batch_size (discriminator reward for each trajectory, applied to each visit of the corresponding
                      trajectory)
            gen_input should be the same as gen_target with origin prepended
        """
        batch_size, seq_len = gen_input.size()
        # seq_len * batch_size
        gen_input = gen_input.permute(1, 0)
        gen_target = gen_target.permute(1, 0)
        h = self.init_hidden(batch_size)
        loss = 0.0
        for i in range(seq_len):
            # the i-th visit of all trajectories in the batch
            pred, h = self.forward(gen_input[i], h)
            for j in range(batch_size):
                # the i-th visit of the j-th trajectory in the batch
                loss += -pred[j][gen_target.data[i][j]] * reward[j]  # log(P(y_t|Y_1:Y_{t-1})) * Q
        loss /= batch_size
        return loss
