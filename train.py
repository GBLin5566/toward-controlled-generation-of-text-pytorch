import time

import numpy as np
from keras.preprocessing import sequence
from keras.datasets import imdb
import torch
from torch.autograd import Variable
from torch.optim import Adam

import Model.Constants as Constants
from Model.Modules import Encoder, Generator, Discriminator
from utils import check_cuda

max_features = 10000
maxlen = 20
batch_size = 128
epoch = 3
c_dim = 2
d_word_vec = 150
lambda_c = 0.1
lambda_z = 0.1
use_cuda = True

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
    input_data = Variable(tensor, volatile=testing, requires_grad=False)
    input_data = check_cuda(input_data, use_cuda)
    output_data = input_data
    return input_data, output_data
    
def get_batch_label(data, label, index, batch_size, testing=False):
    tensor = torch.from_numpy(data[index:index+batch_size]).type(torch.LongTensor)
    input_data = Variable(tensor, volatile=testing, requires_grad=False)
    input_data = check_cuda(input_data, use_cuda)
    label_tensor = torch.from_numpy(label[index:index+batch_size]).type(torch.LongTensor)
    output_data = Variable(label_tensor, volatile=testing, requires_grad=False)
    output_data = check_cuda(output_data, use_cuda)
    return input_data, output_data

# Borrow from https://github.com/ethanluoyc/pytorch-vae/blob/master/vae.py
def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

# Make instances
encoder = Encoder(
        n_src_vocab=max_features, 
        use_cuda=use_cuda,
        )
decoder = Generator(
        n_target_vocab=max_features, 
        c_dim=c_dim,
        use_cuda=use_cuda,
        )
discriminator = Discriminator(
        n_src_vocab=max_features,
        maxlen=maxlen,
        use_cuda=use_cuda,
        )
encoder = check_cuda(encoder, use_cuda)
decoder = check_cuda(decoder, use_cuda)
discriminator = check_cuda(discriminator, use_cuda)
criterion = torch.nn.CrossEntropyLoss()
vae_parameters = list(encoder.parameters()) + list(decoder.parameters())
vae_opt = Adam(vae_parameters)
e_opt = Adam(encoder.parameters())
g_opt = Adam(decoder.parameters())
d_opt = Adam(discriminator.parameters())

def train_discriminator(discriminator):
    # TODO: empirical Shannon entropy
    print_epoch = 0
    for epoch_index in range(epoch):
        for batch, index in enumerate(range(0, len(x_train) - 1, batch_size)):
            discriminator.train()
            input_data, output_data = get_batch_label(x_train, y_train, index, batch_size)
            
            discriminator.zero_grad()

            output = discriminator(input_data)
            loss = criterion(output, output_data)
            loss.backward()
            d_opt.step()
        
            if batch % 25 == 0:
                print("[Discriminator] Epoch {} batch {}'s loss: {}".format(
                    epoch_index, 
                    batch, 
                    loss.data[0],
                    ))
            if print_epoch == epoch_index and print_epoch:
                discriminator.eval()
                print_epoch = epoch_index + 1
                input_data, output_data = get_batch_label(x_test, y_test, 0, len(y_test), testing=True)
                _, predicted = torch.max(discriminator(input_data).data, 1)
                correct = (predicted == torch.from_numpy(y_test)).sum()
                print("[Discriminator] Test accuracy {} %".format(
                    100 * correct / len(y_test)
                    ))

def train_vae(encoder, decoder):
    encoder.train()
    decoder.train()
    for epoch_index in range(epoch):
        for batch, index in enumerate(range(0, len(x_train) - 1, batch_size)):
            total_loss = 0
            start_time = time.time()
            
            input_data, output_data = get_batch(x_train, index, batch_size)
            encoder.zero_grad()
            decoder.zero_grad()
            vae_opt.zero_grad()

            # Considering the data may do not have enough data for batching
            # Init. hidden with len(input_data) instead of batch_size
            enc_hidden = encoder.init_hidden(len(input_data))
            # Input of encoder is a batch of sequence.
            enc_hidden = encoder(input_data, enc_hidden)

            # Generate the random one-hot array from prior p(c)
            # NOTE: Assume general distribution for now
            random_one_dim = np.random.randint(c_dim, size=len(input_data))
            one_hot_array = np.zeros((len(input_data), c_dim))
            one_hot_array[np.arange(len(input_data)), random_one_dim] = 1
            
            c = torch.from_numpy(one_hot_array).float()
            var_c = Variable(c, requires_grad=False)
            var_c = check_cuda(var_c, use_cuda)
            # TODO: use iteration along first dim.
            cat_hidden = (torch.cat([enc_hidden[0][0], var_c], dim=1).unsqueeze(0), 
                    torch.cat([decoder.init_hidden_c_for_lstm(len(input_data))[0], var_c], dim=1).unsqueeze(0))

            # Reshape output_data from (batch_size, seq_len) to (seq_len, batch_size)
            output_data = output_data.permute(1, 0)
            # Input of decoder is a batch of word-by-word.
            for index, word in enumerate(output_data):
                if index == len(output_data) - 1:
                    break
                output, cat_hidden = decoder(word, cat_hidden)
                next_word = output_data[index+1]
                total_loss += criterion(output.view(-1, max_features), next_word)
            # Train
            avg_loss = total_loss.data[0] / maxlen
            ll = latent_loss(encoder.z_mean, encoder.z_sigma)
            total_loss += ll
            total_loss.backward()
            vae_opt.step()

            if batch % 25 == 0:
                print("[VAE] Epoch {} batch {}'s average language loss: {}, latent loss: {}".format(
                    epoch_index, 
                    batch, 
                    avg_loss,
                    ll.data[0],
                    ))

def train_vae_with_attr_loss(encoder, decoder, discriminator):
    for epoch_index in range(epoch):
        for batch, index in enumerate(range(0, len(x_train) - 1, batch_size)):
            encoder.zero_grad()
            decoder.zero_grad()
            vae_loss = 0
            
            input_data, output_data = get_batch_label(x_train, y_train, index, batch_size)
            
            enc_hidden = encoder.init_hidden(len(input_data))
            enc_hidden = encoder(input_data, enc_hidden)
   
            target = np.array([output_data.cpu().data.numpy()]).reshape(-1)
            one_hot_array = np.eye(c_dim)[target]
            
            c = torch.from_numpy(one_hot_array).float()
            var_c = Variable(c, requires_grad=False)
            var_c = check_cuda(var_c, use_cuda)
            # TODO: use iteration along first dim.
            cat_hidden = (torch.cat([enc_hidden[0][0], var_c], dim=1).unsqueeze(0), 
                    torch.cat([decoder.init_hidden_c_for_lstm(len(input_data))[0], var_c], dim=1).unsqueeze(0))

            batch_init_word = np.zeros((batch_size, max_features))
            batch_init_word[np.arange(batch_size), Constants.BOS] = 1
            batch_init_word = Variable(torch.from_numpy(batch_init_word)).float()
            batch_init_word = check_cuda(batch_init_word, use_cuda)

            input_data = input_data.permute(1, 0)
            for index in range(maxlen - 1):
                if 'next_word' in locals():
                    word = next_word.squeeze(1)
                    word = check_cuda(word, use_cuda)
                    output, cat_hidden, pre_soft = decoder(word, cat_hidden, low_temp=True, one_hot_input=True)
                else:
                    word = batch_init_word
                    word = check_cuda(word, use_cuda)
                    output, cat_hidden, pre_soft = decoder(word, cat_hidden, low_temp=True, one_hot_input=True)
                # From one-hot to word embedding
                next_word = output
                correct_word = input_data[index+1]
                vae_loss += criterion(pre_soft.view(-1, max_features), correct_word)
                if len(batch_init_word.size()) == 2:
                    batch_init_word = batch_init_word.unsqueeze(1)
                if len(next_word.size()) == 2:
                    next_word = next_word.unsqueeze(1)
                batch_init_word = torch.cat([batch_init_word, next_word], dim=1)
            # NOTE Latent loss
            ll = latent_loss(encoder.z_mean, encoder.z_sigma)
            # NOTE L_attr_c loss
            generated_sentence = batch_init_word
            discriminator.eval()
            logit = discriminator(generated_sentence, dont_pass_emb=True)
            l_attr_c = criterion(logit, output_data)
            # NOTE L_attr_z loss
            encoder.eval()
            generated_sentence = decoder.one_hot_to_word_emb(generated_sentence)
            encoded_gen = encoder.init_hidden(len(generated_sentence))
            encoded_gen = encoder(generated_sentence, encoded_gen, dont_pass_emb=True)
            l_attr_z = latent_loss(encoder.z_mean, encoder.z_sigma)
            
            avg_loss = vae_loss.data[0] / maxlen

            total_vae_loss = vae_loss + ll
            extra_decoder_loss = lambda_c * l_attr_c + lambda_z * l_attr_z
            total_vae_loss.backward()
            e_opt.step()
            extra_decoder_loss.backward()
            g_opt.step()

            if batch % 25 == 0:
                print("[Attr] Epoch {} batch {}'s average language loss: {}, latent loss: {}".format(
                    epoch_index, 
                    batch, 
                    avg_loss,
                    ll.data[0],
                    ))
                print("l_attr_c loss: {}, l_attr_z loss: {}".format(
                    l_attr_c.data[0],
                    l_attr_z.data[0],
                    ))
def main_alg(encoder, decoder, discriminator):
    train_vae(encoder, decoder)
    repeat_times = 10
    for repeat_index in range(repeat_times):
        train_discriminator(discriminator)
#train_vae(encoder, decoder)
train_vae_with_attr_loss(encoder, decoder, discriminator)
