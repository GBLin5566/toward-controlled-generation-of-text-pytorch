import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    '''A LSTM encoder to encode a sentence into a latent vector z.'''
    def __init__():
        super(Encoder, self).__init__()

    def forward():
        pass

class Generator(nn.Module):
    '''A LSTM generator to synthesis a sentence with input (z, c)
       where z is a latent vector from encoder and c is attribute code.
    '''
    def __init__():
        super(Generator, self).__init__()

    def forward():
        pass

class Discriminator(nn.Module):
    '''A CNN discriminator to classify the attributes given a sentence.'''
    def __init__():
        super(Discriminator, self).__init__()

    def forward():
        pass
