
import time

import numpy as np
from keras.preprocessing import sequence
from keras.datasets import imdb
import torch
from torch.autograd import Variable

import Model.Constants as Constants
from Model.Modules import Encoder, Generator, Discriminator

max_features = 25000
maxlen = 100
batch_size = 64

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=max_features,
        start_char=Constants.BOS,
        oov_char=Constants.UNK,
        index_from=Constants.EOS,
        )

forward_dict = imdb.get_word_index()
for key, value in forward_dict.items():
    forward_dict[key] = value + Constants.EOS
forward_dict[Constants.PAD_WORD] = Constants.PAD
forward_dict[Constants.UNK_WORD] = Constants.UNK
forward_dict[Constants.BOS_WORD] = Constants.BOS
forward_dict[Constants.EOS_WORD] = Constants.EOS

backward_dict = {}
for key, value in forward_dict.items():
    backward_dict[value] = key

x_train = sequence.pad_sequences(
        x_train, 
        maxlen=maxlen,
        padding='post',
        truncating='post',
        value=Constants.PAD,
        )
x_test = sequence.pad_sequences(
        x_test, 
        maxlen=maxlen,
        padding='post',
        truncating='post',
        value=Constants.PAD,
        )

def get_batch(data, index, batch_size, testing=False):
    tensor = torch.from_numpy(data[index:index+batch_size]).type(torch.LongTensor)
    input_data = Variable(tensor, volatile=testing)
    output_data = input_data
    return input_data, output_data

encoder = Encoder(n_src_vocab=max_features)
decoder = Generator(n_target_vocab=max_features)

encoder.train()
decoder.train()
total_loss = 0
start_time = time.time()
enc_hidden = encoder.init_hidden(batch_size)
for batch, index in enumerate(range(0, len(x_train) - 1, batch_size)):
    input_data, output_data = get_batch(x_train, index, batch_size)
    encoder.zero_grad()
    decoder.zero_grad()

    enc_hidden = encoder(input_data, enc_hidden)
    print(enc_hidden)
