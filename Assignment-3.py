# -*- coding: utf-8 -*-
"""Assignment3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YX9iA3MLzmE15rxBMDwKs_D8Vm1KUTNL
"""

train_path = "/content/hin_train.csv"
val_path = "/content/hin_valid.csv"
test_path = "/content/hin_test.csv"

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import zipfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import csv

device = "cuda" if torch.cuda.is_available() else "cpu"


def createVocab(path):
    # d = pd.read_csv(path,sep="\t",header=None)
    # d = d.dropna()
    # print(d.head())

    file = open(path)
    dataset = csv.reader(file, delimiter=",")

    hindi = []
    english = []

    # get the words in a list

    for data in dataset:
        english.append(data[0])
        hindi.append(data[1])

    # print(english)
    # print(hindi)

    # append start and end characters to output - kannada
    for i in range(len(hindi)):
        hindi[i] = "\t" + hindi[i] + "\n"

    return np.array(hindi), np.array(english)


# createVocab(train_path)


def getChar(data):
    data_char = set()  # to store the the different characters present in data
    data_char.add(" ")
    for word in data:
        for char in word:
            if char not in data_char:
                data_char.add(char)

    # sort the characters in dataset
    data_char = sorted(list(data_char))

    # number of characters in the set
    num_tokens = len(data_char)

    # get the max length of the words
    max_len = max([len(word) for word in data])

    # return set of all characters in data
    return data_char, num_tokens, max_len


def getData(
    english,
    hindi,
    max_enc_len,
    max_dec_len,
    num_decoder_tokens,
    input_token_index,
    target_token_index,
):
    # initializing with 0s for max_length
    encoder_input_data = np.zeros(
        (len(english), max_enc_len), dtype="float32"
    )  # (51200,24)
    decoder_input_data = np.zeros(
        (len(english), max_dec_len), dtype="float32"
    )  # (51200,22)
    decoder_target_data = np.zeros(
        (len(english), max_dec_len, len(target_token_index)), dtype="float32"
    )  # (51200,22,67)

    # populating indices for characters that exist
    for i, (english, hindi) in enumerate(zip(english, hindi)):
        for t, char in enumerate(english):
            encoder_input_data[i, t] = input_token_index[char]

        for t, char in enumerate(hindi):
            decoder_input_data[i, t] = target_token_index[char]
            if t > 0:
                # decoder_target_data will be ahead by one timestep and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

        decoder_input_data[i, t + 1 :] = target_token_index[" "]
        decoder_target_data[i, t:, target_token_index[" "]] = 1.0

    return encoder_input_data, decoder_input_data, decoder_target_data


def createDictionary(input_tokens, target_tokens):
    input_token_index = dict([(char, i) for i, char in enumerate(input_tokens)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_tokens)])

    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items()
    )
    return (
        input_token_index,
        target_token_index,
        reverse_input_char_index,
        reverse_target_char_index,
    )


def main():
    # get the training words as an array from train directory
    train_hindi_words, train_english_words = createVocab(train_path)
    # get validation words
    val_hindi_words, val_english_words = createVocab(val_path)

    test_hindi_words, test_english_words = createVocab(test_path)

    # get the characters from train and val dataset
    train_eng_characters, train_num_encoder_tokens, train_max_enc_len = getChar(
        train_english_words
    )
    train_hin_characters, train_num_decoder_tokens, train_max_dec_len = getChar(
        train_hindi_words
    )

    val_eng_characters, val_num_encoder_tokens, val_max_enc_len = getChar(
        val_english_words
    )
    val_hin_characters, val_num_decoder_tokens, val_max_dec_len = getChar(
        val_hindi_words
    )

    test_eng_characters, test_num_encoder_tokens, test_max_enc_len = getChar(
        train_english_words
    )
    test_hin_characters, test_num_decoder_tokens, test_max_dec_len = getChar(
        test_hindi_words
    )

    # take the largest number of tokens and max_length of words on both encoder and decoder
    num_encoder_tokens = max(
        val_num_encoder_tokens, train_num_encoder_tokens, test_num_encoder_tokens
    )
    num_decoder_tokens = max(
        val_num_decoder_tokens, train_num_decoder_tokens, test_num_decoder_tokens
    )

    max_enc_len = max(train_max_enc_len, val_max_enc_len, test_max_enc_len)
    max_dec_len = max(train_max_dec_len, val_max_dec_len, test_max_dec_len)
    # print(max_enc_len)
    # print(max_dec_len)

    input_hin_characters = set()
    for char in train_hin_characters:
        input_hin_characters.add(char)

    for char in val_hin_characters:
        input_hin_characters.add(char)

    for char in test_hin_characters:
        input_hin_characters.add(char)

    # print(len(input_hin_characters))
    # print(input_hin_characters)
    # making a dictionary and reverse dictionary to map the characters with the indices and indices to characters
    (
        input_token_index,
        target_token_index,
        reverse_input_char_index,
        reverse_target_char_index,
    ) = createDictionary(train_eng_characters, list(input_hin_characters))

    # print(len(input_token_index))
    # print(len(target_token_index))

    (
        train_encoder_input_data,
        train_decoder_input_data,
        train_decoder_target_data,
    ) = getData(
        train_english_words,
        train_hindi_words,
        max_enc_len,
        max_dec_len,
        num_decoder_tokens,
        input_token_index,
        target_token_index,
    )
    val_encoder_input_data, val_decoder_input_data, val_decoder_target_data = getData(
        val_english_words,
        val_hindi_words,
        max_enc_len,
        max_dec_len,
        num_decoder_tokens,
        input_token_index,
        target_token_index,
    )

    train_encoder_input_data = (
        torch.from_numpy(train_encoder_input_data).to(device).long()
    )

    train_decoder_input_data = (
        torch.from_numpy(train_decoder_input_data).to(device).long()
    )
    train_decoder_target_data = (
        torch.from_numpy(train_decoder_target_data).to(device).long()
    )

    # print(train_encoder_input_data.size())
    # print(train_decoder_input_data.size())
    # print(train_decoder_target_data.size())
    return (
        train_encoder_input_data,
        train_decoder_input_data,
        train_decoder_target_data,
        max_enc_len,
        max_dec_len,
        len(input_token_index),
        len(target_token_index),
    )


# max_enc_len=24
# max_dec_len=22
# input_token_index=27
# target_token_index=67

# main()


class Encoder(nn.Module):
    def __init__(
        self,
        train_eng_characters,
        embedding_size,
        hidden_size,
        input_token_index_len,
        no_of_layers,
    ):
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.train_eng_characters = train_eng_characters
        self.hidden_size = hidden_size
        self.input_token_index_len = input_token_index_len
        self.no_of_layers = no_of_layers
        self.encoder_embedding = nn.Embedding(
            self.input_token_index_len, self.embedding_size
        ).to(device)
        self.encoder_rnn = nn.GRU(
            self.embedding_size, hidden_size, self.no_of_layers, batch_first=True
        ).to(device)

    def forward(self, input, hidden):
        enc_embedd = self.encoder_embedding(input)
        out, enc_hidden = self.encoder_rnn(enc_embedd, hidden)
        return out, enc_hidden


class Decoder(nn.Module):
    def __init__(
        self,
        train_hin_characters,
        embedding_size,
        hidden_size,
        target_token_index_len,
        no_of_layers,
    ):
        super(Decoder, self).__init__()
        self.embedding_size = embedding_size
        self.train_hin_characters = train_hin_characters
        self.hidden_size = hidden_size
        self.no_of_layers = no_of_layers
        self.target_token_index_len = target_token_index_len
        self.decoder_embedding = nn.Embedding(
            self.target_token_index_len, self.embedding_size
        ).to(device)
        self.decoder_rnn = nn.GRU(
            self.embedding_size, self.hidden_size, self.no_of_layers, batch_first=True
        ).to(device)
        self.linear = nn.Linear(
            self.hidden_size, self.target_token_index_len, bias=True
        ).to(device)
        # dim = 2
        self.softmax = nn.Softmax(dim=2).to(device)

    def forward(self, input, hidden):
        dec_embedd = self.decoder_embedding(input)
        out, dec_hidden = self.decoder_rnn(dec_embedd, hidden)
        output1 = self.linear(out)
        return output1, dec_hidden
        output2 = self.softmax(output1)
        return output2, hidden1


hidden_size = 256

embedding_size = 256

no_of_layers = 2

epochs = 20

batchsize = 1024


def accuracy(target, predictions, flag):
    total = 0
    for x in range(len(target)):
        if torch.equal(target[x], predictions[x]):
            total += 1
    return total


def train():
    torch.autograd.set_detect_anomaly(True)
    (
        encoder_input_data,
        decoder_input_data,
        decoder_target_data,
        max_enc_len,
        max_dec_len,
        input_token_index_len,
        target_token_index_len,
    ) = main()
    encoder = Encoder(
        encoder_input_data,
        embedding_size,
        hidden_size,
        input_token_index_len,
        no_of_layers,
    ).to(device)
    decoder = Decoder(
        decoder_input_data,
        embedding_size,
        hidden_size,
        target_token_index_len,
        no_of_layers,
    ).to(device)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    for _ in range(epochs):
        total_loss = 0
        total_acc = 0
        for x in range(0, len(encoder_input_data), batchsize):
            loss = 0
            input_tensor = encoder_input_data[x : x + batchsize].to(device)
            hidden_input0 = torch.zeros(no_of_layers, batchsize, hidden_size).to(device)
            if input_tensor.size()[0] < batchsize:
                break
            output, hidden = encoder.forward(input_tensor, hidden_input0)
            input2 = (
                decoder_input_data[x : x + batchsize, 0].to(device).resize(batchsize, 1)
            )
            # input2 = (torch.tensor(input2)).view(batchsize,1).to(device)
            hidden1 = hidden
            predicted = []
            predictions = []
            for i in range(22):
                output1, hidden1 = decoder.forward(input2, hidden1)
                # print(output1.size()) #(1024,1,68)
                # predicted.append(output1)
                # print(len(predicted[0]))
                output2 = decoder.softmax(output1)
                predicted.append(output2)
                # print(output2.size()) #(1024,1,68)
                output3 = torch.argmax(output2, dim=2)
                predictions.append(output3)
                input2 = output3

            predicted = (
                torch.cat(tuple(x for x in predicted), dim=1)
                .to(device)
                .resize(max_dec_len * batchsize, target_token_index_len)
            )
            predictions = torch.cat(tuple(x for x in predictions), dim=1).to(device)
            total_acc += accuracy(
                decoder_target_data[x : x + batchsize].to(device), predictions, x
            )
            loss = nn.CrossEntropyLoss(reduction="sum")(
                predicted, decoder_input_data[x : x + batchsize].reshape(-1).to(device)
            )
            with torch.no_grad():
                total_loss += loss.item()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1)
            encoder_optimizer.step()
            decoder_optimizer.step()
        print(total_loss / (51200 * 22))
        print(total_acc)


def train_setup(
    net,
    lr=0.01,
    n_batches=100,
    batch_size=10,
    momentum=0.9,
    display_freq=5,
    device="cpu",
):
    net = net.to(device)
    criterion = nn.NLLLoss(ignore_index=-1)
    opt = optim.Adam(net.parameters(), lr=lr)
    teacher_force_upto = n_batches // 3

    loss_arr = np.zeros(n_batches + 1)

    for i in range(n_batches):
        loss_arr[i + 1] = (
            loss_arr[i] * i
            + train_batch(
                net,
                opt,
                criterion,
                batch_size,
                device=device,
                teacher_force=i < teacher_force_upto,
            )
        ) / (i + 1)

        if i % display_freq == display_freq - 1:
            clear_output(wait=True)

            print("Iteration", i, "Loss", loss_arr[i])
            plt.figure()
            plt.plot(loss_arr[1:i], "-*")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.show()
            print("\n\n")

    torch.save(net, "model.pt")
    return loss_arr