import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

import Model.Constants as Constants
from utils import check_cuda

class Encoder(nn.Module):
    '''A LSTM encoder to encode a sentence into a latent vector z.'''
    def __init__(
            self,
            n_src_vocab,
            n_layers=1,
            d_word_vec=150,
            d_inner_hid=300,
            dropout=0.1,
            d_out_hid=300,
            use_cuda=False,
            ):
        super(Encoder, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.n_layers = n_layers
        self.d_inner_hid = d_inner_hid
        self.use_cuda = use_cuda

        # NOTE:Maybe try GRU
        self.rnn = nn.LSTM(d_word_vec, d_inner_hid, n_layers, dropout=dropout)

        # For generating Gaussian distribution
        self._enc_mu = nn.Linear(d_inner_hid, d_out_hid)
        self._enc_log_sigma = nn.Linear(d_inner_hid, d_out_hid)

        self.init_weights()

    # Borrow from https://github.com/ethanluoyc/pytorch-vae/blob/master/vae.py
    def _sample_latent(self, enc_hidden):
        mu = self._enc_mu(enc_hidden)
        log_sigma = self._enc_log_sigma(enc_hidden)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        std_z_var = Variable(std_z, requires_grad=False)
        std_z_var = check_cuda(std_z_var, self.use_cuda)
        return mu + sigma * Variable(std_z, requires_grad=False)
    
    def forward(self, src_seq, hidden):
        enc_input = self.drop(self.src_word_emb(src_seq))
        # Reshape tensor's shape to (d_word_vec, batch_size, d_inner_hid)
        enc_input = enc_input.permute(1, 0, 2)
        _, hidden = self.rnn(enc_input, hidden)
        hidden = (
                self._sample_latent(hidden[0]), 
                hidden[1]
                )
        return hidden

    def init_hidden(self, batch_size):
        # NOTE: LSTM needs 2 hidden states
        hidden = [
            Variable(torch.zeros(self.n_layers, batch_size, self.d_inner_hid)),
            Variable(torch.zeros(self.n_layers, batch_size, self.d_inner_hid))
            ]
        hidden[0] = check_cuda(hidden[0], self.use_cuda)
        hidden[1] = check_cuda(hidden[1], self.use_cuda)
        return hidden

    def init_weights(self):
        initrange = 0.1
        self.src_word_emb.weight.data.uniform_(-initrange, initrange)
        self._enc_mu.weight.data.uniform_(-initrange, initrange)
        self._enc_log_sigma.weight.data.uniform_(-initrange, initrange)

class Generator(nn.Module):
    '''A LSTM generator to synthesis a sentence with input (z, c)
       where z is a latent vector from encoder and c is attribute code.
    '''
    def __init__(
            self,
            n_target_vocab,
            n_layers=1,
            d_word_vec=150,
            d_inner_hid=300,
            c_dim=1,
            dropout=0.1,
            use_cuda=False,
            ):
        super(Generator, self).__init__()

        self.drop = nn.Dropout(dropout)

        self.d_inner_hid = d_inner_hid
        self.c_dim = c_dim
        self.n_layers = n_layers
        self.use_cuda = use_cuda

        self.target_word_emb = nn.Embedding(
            n_target_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.rnn = nn.LSTM(d_word_vec, d_inner_hid + c_dim, n_layers, dropout=dropout)

        self.to_word_emb = nn.Sequential(
                nn.Linear(d_inner_hid + c_dim, d_word_vec),
                nn.ReLU()
                )
        self.linear = nn.Linear(d_word_vec, n_target_vocab)

        self.softmax = nn.LogSoftmax()

        self.init_weights()

    def forward(self, target_word, hidden, low_temp=False):
        ''' hidden is composed of z and c '''
        ''' input is word-by-word in Generator '''
        dec_input = self.drop(
            self.target_word_emb(target_word)).unsqueeze(0)
        output, hidden = self.rnn(dec_input, hidden)
        output = self.to_word_emb(output)
        output = self.linear(output)
        # Low temperature factor trick
        if low_temp:
            output = self.softmax(output / 0.001)
        return output, hidden

    def init_hidden_c_for_lstm(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers, batch_size, self.d_inner_hid))
        hidden = check_cuda(hidden, self.use_cuda)
        return hidden

    def init_weights(self):
        initrange = 0.1
        self.target_word_emb.weight.data.uniform_(-initrange, initrange)
        self.to_word_emb[0].weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)

class Discriminator(nn.Module):
    '''A CNN discriminator to classify the attributes given a sentence.'''
    def __init__(
            self,
            n_src_vocab,
            d_word_vec=300,
            dropout=0.1,
            use_cuda=False,
            ):
        super(Discriminator, self).__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.drop = nn.Dropout(dropout)
        self.conv = []
        self.filter_size = [3, 4, 5]
        self.conv1 = nn.Conv1d(500, 128, kernel_size=5)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=5)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5)
        self.linear = nn.Linear(3968, 2)

    def forward(self, input_sentence):
        emb_sentence = self.src_word_emb(input_sentence)
        relu1 = F.relu(self.conv1(emb_sentence))
        layer1 = F.max_pool1d(relu1, 3)
        relu2 = F.relu(self.conv2(layer1))
        layer2 = F.max_pool1d(relu2, 3)
        layer3 = F.max_pool1d(F.relu(self.conv2(layer2)), 10)
        flatten = self.drop(layer2.view(layer3.size()[0], -1))
        logit = self.linear(flatten)
        return logit
