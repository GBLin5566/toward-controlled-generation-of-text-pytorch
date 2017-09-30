import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import Model.Constants as Constants

is_cuda = torch.cuda.is_available()

class Encoder(nn.Module):
    '''A LSTM encoder to encode a sentence into a latent vector z.'''
    def __init__(
            self,
            n_src_vocab,
            n_layers=1,
            d_word_vec=300,
            d_inner_hid=300,
            dropout=0.1,
            d_out_hid=300,
            ):
        super(Encoder, self).__init__()

        self.drop = nn.Dropout(dropout)

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        # NOTE:Maybe try GRU
        self.rnn = nn.LSTM(d_word_vec, d_inner_hid, n_layers, dropout=dropout)

        # For generating Gaussian distribution
        self._enc_mu = nn.Linear(encoder_hid, d_out_hid)
        self._enc_log_sigma = nn.Linear(encoder_hid, d_out_hid)

    def _sample_latent(self, enc_hidden):
        mu = self._enc_mu(enc_hidden)
        log_sigma = self._enc_log_sigma(enc_hidden)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)
    
    def forward(self, src_seq, hidden, return_z=False):
        enc_input = self.drop(self.src_word_emb(src_seq))
        _, hidden = self.rnn(enc_input, hidden)
        if return_z:
            hidden = self._sample_latent(hidden)
        return hidden

    def init_hidden(self, batch_size):
        # NOTE: LSTM needs 2 hidden states
        hidden = (
            Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
            Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
            )
        if is_cuda:
            hidden[0] = hidden[0].cuda()
            hidden[1] = hidden[1].cuda()
        return hidden

class Generator(nn.Module):
    '''A LSTM generator to synthesis a sentence with input (z, c)
       where z is a latent vector from encoder and c is attribute code.
    '''
    def __init__(
            self,
            n_target_vocab,
            n_layers=1,
            d_word_vec=300,
            d_inner_hid=300,
            c_dim=1,
            dropout=0.1,
            ):
        super(Generator, self).__init__()

        self.drop = nn.Dropout(dropout)

        self.d_inner_hid = d_inner_hid
        slef.c_dim = c_dim

        self.target_word_emb = nn.Embedding(
            n_target_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.rnn = nn.LSTM(d_word_vec, d_inner_hid + c_dim, n_layers, dropout=dropout)

        self.linear = nn.Linear(d_word_vec, n_target_vocab)

        self.softmax = nn.LogSoftmax()

    def forward(self, target_word_emb, hidden):
        dec_input = self.drop(
            self.target_word_emb(target_word_emb))
        output, hidden = self.rnn(dec_input, hidden)
        output = F.relu(self.linear(output))
        output = self.softmax(output)
        
        return output, hidden

    def init_hidden(self, batch_size):
        # NOTE: LSTM needs 2 hidden states
        hidden = (
            Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
            Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
            )
        if is_cuda:
            hidden[0] = hidden[0].cuda()
            hidden[1] = hidden[1].cuda()
        return hidden

class Discriminator(nn.Module):
    '''A CNN discriminator to classify the attributes given a sentence.'''
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self):
        pass
