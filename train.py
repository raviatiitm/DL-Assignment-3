# importing libraries
import torch
import pandas as pd
import numpy as np
import wandb
import matplotlib.pyplot as plt
import os
import torch
import random
import torchvision
import zipfile
import argparse
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision import transforms 
import torch.nn as nn
import torch.optim as optim
import csv
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='sequence-to-sequence learning')
parser.add_argument('-wp','--wandb_project', default="DL-Assignment3", required=False,metavar="", type=str, help=' ')
parser.add_argument('-we','--wandb_entity', default="cs22m069", required=False,metavar="", type=str, help='')
parser.add_argument('-e','--epochs', default=10, required=False,metavar="", type=int, help=' ')
parser.add_argument('-b','--batchsize', default=1024, required=False,metavar="", type=int, help=' ')
parser.add_argument('-hidden','--hidden_size', default=128, required=False,metavar="", type=int, help=' ')
parser.add_argument('-embed','--embedding_size',default = 128,required=False,metavar="", type=int, help=' ')
parser.add_argument('-cell','--cell_type',default = 'LSTM',required=False,metavar="", type=str, help=' ',choices = ['LSTM','GRU','RNN'])
parser.add_argument('-drop','--dropout',default = 0.3,required=False,metavar="", type=float, help=' ',choices = [0.1,0.2,0.3,0.4,0.5])
parser.add_argument('-attn','--attentionRequired',default = True,required=False,metavar="", type=bool, help=' ',choices=[True,False])
parser.add_argument('-layer','--no_of_layers',default = 1,required=False,metavar="", type=int, help=' ')
args = parser.parse_args()

def getWords(path):
        
        hindi = []
        english = []

        file = open(path)
        dataset = csv.reader(file, delimiter = ",")

        # to get the words in a list

        for data in dataset:
          e=data[0]
          h=data[1]
          english.append(e)
          hindi.append(h)

        #appending the start and end characters to hindi words
        for i in range(len(hindi)):
            hindi[i] = "\t" + hindi[i] +"\n"

        return np.array(hindi), np.array(english)

def getChar(data):
      data_char = set() #to store the the unique characters present in data
      data_char.add(" ")
      for word in data:
        for char in word:
          if char not in data_char:
            data_char.add(char)

      #sort the characters in dataset
      data_char = sorted(list(data_char))

      #number of characters in the set
      num_tokens = len(data_char)

      #get the max length of the words
      max_len = max([len(word) for word in data])

      #return set of all characters in data
      return data_char, num_tokens, max_len

def getData(english, hindi, max_enc_len, max_dec_len, num_decoder_tokens, input_token_index, target_token_index):
      #initializing with 0s for max_length
      encoder_input_data = np.zeros((len(english), max_enc_len), dtype="float32") 
      decoder_input_data = np.zeros((len(english), max_dec_len), dtype="float32") 

      #creating indices for characters that exist
      for i, (english, hindi) in enumerate(zip(english, hindi)):
          for t, char in enumerate(english):
              encoder_input_data[i, t] = input_token_index[char]
          
          for t, char in enumerate(hindi):
              decoder_input_data[i, t] = target_token_index[char]       
          decoder_input_data[i, t+1:] = target_token_index[' ']

      return encoder_input_data, decoder_input_data

def createDictionary(input_tokens, target_tokens):
    #making a dictionary to map the characters with the indices
    input_token_index = dict([(char, i) for i, char in enumerate(input_tokens)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_tokens)])
    
    #making a dictionary to map the indices with the characters
    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())
    return input_token_index,target_token_index,reverse_input_char_index,reverse_target_char_index


#get the training words as an array from train data
train_hindi_words, train_english_words =getWords('/content/hin_train.csv')
#get validation words
val_hindi_words, val_english_words =getWords('/content/hin_valid.csv')
#get test words
test_hindi_words, test_english_words =getWords('/content/hin_test.csv')
  
#get the characters from train and val and test dataset
train_eng_characters, train_num_encoder_tokens, train_max_enc_len = getChar(train_english_words)
train_hin_characters, train_num_decoder_tokens, train_max_dec_len = getChar(train_hindi_words)

val_eng_characters, val_num_encoder_tokens, val_max_enc_len = getChar(val_english_words)
val_hin_characters, val_num_decoder_tokens, val_max_dec_len = getChar(val_hindi_words)

test_eng_characters, test_num_encoder_tokens, test_max_enc_len = getChar(test_english_words)
test_hin_characters, test_num_decoder_tokens, test_max_dec_len = getChar(test_hindi_words)

#take the largest number of tokens and max_length of words on both encoder and decoder
num_encoder_tokens = max(val_num_encoder_tokens, train_num_encoder_tokens,test_num_encoder_tokens)
num_decoder_tokens = max(val_num_decoder_tokens, train_num_decoder_tokens,test_num_decoder_tokens)

max_enc_len = max(train_max_enc_len, val_max_enc_len,test_max_enc_len)
max_dec_len = max(train_max_dec_len, val_max_dec_len,test_max_dec_len)

#finding the set having unique characters from train, val,test data
input_hin_characters=set()
for char in train_hin_characters:
  input_hin_characters.add(char)
        
for char in val_hin_characters:
  input_hin_characters.add(char)
        
for char in test_hin_characters:
  input_hin_characters.add(char)
        

# calling createDictionary function to make a dictionary and reverse dictionary to map the characters with the indices and indices to characters
input_token_index,target_token_index,reverse_input_char_index,reverse_target_char_index=createDictionary(train_eng_characters,list(input_hin_characters)) 

#creating input data for encoder and decoder to process in batch
train_encoder_input_data, train_decoder_input_data = getData(train_english_words, train_hindi_words, max_enc_len, max_dec_len, num_decoder_tokens, input_token_index, target_token_index)
val_encoder_input_data, val_decoder_input_data = getData(val_english_words, val_hindi_words, max_enc_len, max_dec_len, num_decoder_tokens, input_token_index, target_token_index)
test_encoder_input_data, test_decoder_input_data = getData(test_english_words, test_hindi_words, max_enc_len, max_dec_len, num_decoder_tokens, input_token_index, target_token_index)

# converting to tensors from numpy
train_encoder_input_data=torch.from_numpy(train_encoder_input_data).to(device).long()
train_decoder_input_data=torch.from_numpy(train_decoder_input_data).to(device).long()

val_encoder_input_data=torch.from_numpy(val_encoder_input_data).to(device).long()
val_decoder_input_data=torch.from_numpy(val_decoder_input_data).to(device).long()

test_encoder_input_data=torch.from_numpy(test_encoder_input_data).to(device).long()
test_decoder_input_data=torch.from_numpy(test_decoder_input_data).to(device).long()

"""##**Encoder Part**"""

class Encoder(nn.Module):
    def __init__(self,cell,embedding_size,hidden_size,no_of_layers,dropout):
        super(Encoder,self).__init__()
        #setting the encoder part
        self.hidden_size=hidden_size
        self.input_token_index_len=len(input_token_index)
        self.no_of_layers = no_of_layers
        self.embedding_size = embedding_size
        self.cell=cell
        self.train_eng_characters=train_eng_characters
        self.drop = nn.Dropout(dropout) # using dropout
        #creating an embedding layer
        self.encoder_embedding = nn.Embedding(self.input_token_index_len,self.embedding_size).to(device)
        self.gru = nn.GRU(self.embedding_size,self.hidden_size,self.no_of_layers,batch_first = True,bidirectional = True).to(device)
        self.rnn = nn.RNN(self.embedding_size,self.hidden_size,self.no_of_layers,batch_first = True,bidirectional = True).to(device)
        self.lstm = nn.LSTM(self.embedding_size,self.hidden_size,self.no_of_layers,batch_first = True,bidirectional = True).to(device)
    
    #forward function for encoder
    def forward(self,input,hidden,cell):
        enc_embedd= self.encoder_embedding(input)
        temp=enc_embedd
        #using dropout
        enc_embedd = self.drop(temp)
        # RNN/GRU/LSTM layer of the encoder
        if(self.cell == 'RNN'):
            input1=enc_embedd
            input2=hidden
            output,hidden = self.rnn(input1,input2)
        elif(self.cell == 'GRU'):
            input1=enc_embedd
            input2=hidden
            output,hidden = self.gru(input1,input2)
        elif(self.cell == 'LSTM'):
            input1=enc_embedd
            output,(hidden,cell) = self.lstm(input1,(hidden,cell))
        return output,(hidden,cell)

"""## **Decoder Attention Part**"""

class Attention(nn.Module):
    def __init__(self,cell,embedding_size,hidden_size,no_of_layers,dropout,batchsize):
        super(Attention,self).__init__()
        self.layer = no_of_layers
        self.hidden_size = hidden_size
        self.batchsize = batchsize
        self.cell=cell
        self.embedding = nn.Embedding(target_token_index_len,embedding_size).to(device)
        self.drop = nn.Dropout(dropout)
        self.embedding.weight.requires_grad = True
        #calculating lstm model
        input1=embedding_size + hidden_size
        self.LSTM = nn.LSTM(input1,hidden_size,self.layer,batch_first = True).to(device)
        input2=embedding_size + hidden_size
        self.GRU = nn.GRU(input2,hidden_size,self.layer,batch_first = True).to(device)
        input3=embedding_size + hidden_size
        self.RNN = nn.RNN(input3,hidden_size,self.layer,batch_first = True).to(device)
        temp=1
        self.V = nn.Linear(hidden_size,temp,bias = False).to(device)
        temp=hidden_size
        self.U = nn.Linear(hidden_size,temp,bias = False).to(device)
        temp=hidden_size
        self.W = nn.Linear(hidden_size,temp,bias = False).to(device)
        temp=target_token_index_len
        self.linear = nn.Linear(hidden_size,temp,bias=True).to(device)
        self.softmax = nn.Softmax(dim = 2).to(device)
    def forward(self,input,hidden,cell,encoder_outputs):
        ct = torch.zeros(self.batchsize,1,self.hidden_size).to(device)
        Wh=self.W(hidden[-1]).reshape_(self.batchsize,1,self.hidden_size)
        Us=self.U(encoder_outputs)
        temp =  Us+ Wh
        ejt = self.V(torch.tanh(temp))
        alphajt = nn.Softmax(dim = 1)(ejt)
        embedded = self.embedding(input)
        ct = torch.bmm(alphajt.transpose(1,2),encoder_outputs)
        new_input = torch.cat((embedded,ct),dim = 2)
        if(self.rnn == 'LSTM'):
            input1=new_input
            input2=(hidden,cell)
            output,(hidden,cell) = self.LSTM(input1,input2)
        elif(self.rnn == 'RNN'):
            input1=new_input
            input2=hidden
            output,hidden = self.RNN(input1,input2)
        elif(self.rnn == 'GRU'):
            input1=new_input
            input2=hidden
            output,hidden = self.GRU(input1,input2)
        output = self.linear(output)
        temp=(hidden,cell)
        return output,temp

"""## **Helper Functions**"""

#function to get batchsize of data
def getBatchData(input,batchsize,x):
  return input[x:x+batchsize]

# function to do split as output is also biredictional output as encoder is bidirectional. So we are splitting it.
def getSplit(output,hidden_size):
  temp=torch.split(output,[hidden_size,hidden_size],dim = 2)
  input1=temp[0]
  input2=temp[1]
  temp=torch.add(input1,input2)/2
  return temp
  
# function to do resize as encoder is bidirectional asnd decoder is unidirectional. so we are reshaping it.
def getResize(var,val,no_of_layers,batchsize,hidden_size):
  var=var.reshape(2,no_of_layers,batchsize,hidden_size)
  var=torch.add(var[0],var[1])/2
  return var

#function to concatenate and return as tuple
def getConcatenation(lst):
  temp= torch.cat(tuple(x for x in lst),dim =1).to(device)
  return temp

#function to calculate accuracy
def getAccuracy(target,output):
    n=len(target)
    total = 0
    i=0
    while i<n:
        if(torch.equal(target[i],output[i])):
            total += 1
        i+=1
    return total


#function to concatenate characters
def getword(characters):
    return "".join(characters)

#function to get prediction vanilla 
def getPredictions(target,output,df):
    l = len(df)
    n=len(output)
    k=0
    while k<n :
        y_org = []
        y_pred = []
        for y in target[k]:
            if(y == 1):
                break
            else:
                y_org.append(y)
        for y in output[k]:
            if(y == 1):
                break
            else:
                y_pred.append(y)
        df.loc[l,['True']] = getword([reverse_target_char_index[k.item()] for x in y_org])
        df.loc[l,['Predicted']] = getword([reverse_target_char_index[k.item()] for x in y_pred])
        l+=1
        k+=1
    return df

"""## **Function for Test data Evaluation**"""

def Evaluate(attention,test_encoder_input_data,test_decoder_input_data,encoder,decoder,batchsize,hidden_size,embedding_size,no_of_layers):
        with torch.no_grad():
            total_loss=0
            total_acc=0
            #initializing with 0's
            temp=torch.zeros(2*no_of_layers,batchsize,hidden_size).to(device)
            enc_hidden = temp
            enc_cell = temp
            df = pd.DataFrame() # getiing empty pandas dataframe 
            for x in range(0,len(test_encoder_input_data),batchsize):
                loss=0
                y_pred = [] # to store predicted value by decoder to calculate loss
                y_actual = [] #to store index of predicted value by decoder to calculate accuracy
                input_tensor = getBatchData(test_encoder_input_data,batchsize,x).to(device) # initial input to first hidden state(S0)
                output,(hidden,cell) = encoder.forward(input_tensor,enc_hidden,enc_cell)
                output = getSplit(output,hidden_size)
                input =test_decoder_input_data[x:x+batchsize,0].to(device).reshape(batchsize,1) # to get batchsize of train data for encoder
                #calling resize because encoder is bidirectional
                hidden = getResize(hidden,2,no_of_layers,batchsize,hidden_size)
                cell = getResize(cell,2,no_of_layers,batchsize,hidden_size)
                hiddenj = hidden
                temp1=output if attention else hiddenj
                k=0
                #runing while loop for max no of characters in decoder data
                while k<max_dec_len:
                    output1,(hidden,cell) = decoder.forward(input,hidden,cell,temp1)
                    output2 = decoder.softmax(output1)
                    output3 = torch.argmax(output2,dim = 2)
                    y_pred.append(output1) # storing decoder predicted value
                    y_actual.append(output3) # storing decoder predicted value index
                    input = output3
                    k+=1
                #flattening the tensor
                y_pred = getConcatenation(y_pred).reshape(max_dec_len*batchsize,len(target_token_index))
                y_actual = getConcatenation(y_actual)
                var=test_decoder_input_data[x:x+batchsize]
                total_acc += getAccuracy(var.to(device),y_actual) #calling accuracy function to get accuracy
                input1=test_decoder_input_data[x:x+batchsize]
                df = getPredictions(input1,y_actual,df) #calling getPrediction function to generate prediction vanilla
                #calculating loss
                loss  = nn.CrossEntropyLoss(reduction = 'sum')(y_pred,var.reshape(-1).to(device))
                with torch.no_grad():
                    total_loss += loss.item()
            denom=(len(test_decoder_input_data)*max_dec_len)
            test_loss = total_loss/denom # calculating test loss
            total_loss=0
            #calculating test accuracy
            test_accuracy = (total_acc/len(test_decoder_input_data))
            test_accuracy=test_accuracy*100
            total_acc=0
            return test_loss,test_accuracy,df

"""##**Function for validation data** """

def getValidation(attention,val_encoder_input_data,val_decoder_input_data,encoder,decoder,batchsize,hidden_size,embedding_size,no_of_layers):
    with torch.no_grad():
        temp=torch.zeros(2*no_of_layers,batchsize,hidden_size).to(device)
        enc_hidden = temp
        enc_cell = temp
        total_loss=0
        total_acc=0
        for x in range(0,len(val_encoder_input_data),batchsize):
            y_pred=[]
            y_actual=[]
            loss=0
            #input for encoder
            input_tensor = getBatchData(val_encoder_input_data,batchsize,x).to(device)
            #calling encoder forward function
            output,(hidden,cell) = encoder.forward(input_tensor,enc_hidden,enc_cell)
            output = getSplit(output,hidden_size)
            #input for first hidden state in decoder
            input =val_decoder_input_data[x:x+batchsize,0].to(device).reshape(batchsize,1)
            #calling resize because encoder is bidirectional
            hidden = getResize(hidden,2,no_of_layers,batchsize,hidden_size)
            cell = getResize(cell,2,no_of_layers,batchsize,hidden_size)
            hiddenj = hidden
            k=0
            temp1=output if attention else hiddenj
            #ruuning while loop for max no of characters in decoder data
            while k<max_dec_len:
                output1,(hidden,cell) = decoder.forward(input,hidden,cell,temp1)
                output2 = decoder.softmax(output1)
                output3 = torch.argmax(output2,dim = 2)
                y_pred.append(output1) # storing decoder predicted value
                y_actual.append(output3)# storing decoder predicted value index
                input = output3
                k+=1
            #flattening the tensor
            y_pred = getConcatenation(y_pred).reshape(max_dec_len*batchsize,len(target_token_index))
            y_actual = getConcatenation(y_actual)
            var=val_decoder_input_data[x:x+batchsize]
            total_acc += getAccuracy(var.to(device),y_actual) #calling accuracy function to get accuracy
            #calculating loss
            loss  = nn.CrossEntropyLoss(reduction = 'sum')(y_pred,var.reshape(-1).to(device))
            with torch.no_grad():
                total_loss += loss.item()
        denom=(len(val_decoder_input_data)*max_dec_len)
        val_loss = total_loss/denom  # calculating val loss
        total_loss=0
        #calculating val accuracy
        val_accuracy = (total_acc/len(val_decoder_input_data))
        val_accuracy=val_accuracy*100
        total_acc=0
        return val_loss,val_accuracy

"""##**Train function for Attention**"""

def attentiontrain(rnn,batchsize,hidden_size,embedding_size,no_of_layers,dropout,epochs):
    teacher_ratio = 0.5
    total_loss=0
    total_acc=0
    learning_rate=0.001
    #calling encoder and attention 
    encoder = Encoder(rnn,embedding_size,hidden_size,no_of_layers,dropout).to(device)
    decoder = Attention(rnn,embedding_size,hidden_size,no_of_layers,dropout,batchsize).to(device)
    opt_encoder = optim.Adam(encoder.parameters(),learning_rate)
    opt_decoder  = optim.Adam(decoder.parameters(),learning_rate)
    loss=0
    #running for loop epochs no of times
    for i in range(epochs):
        print('Epoch---',i+1,end=" ")
        for x in range(0,len(train_encoder_input_data),batchsize):
            y_pred = []# to store predicted value by decoder to calculate loss
            y_val = [] #to store index of predicted value by decoder to calculate accuracy
            #initially initializing with zeroes
            hidden_input = torch.zeros(no_of_layers,batchsize,hidden_size).to(device)
            temp=torch.zeros(2*no_of_layers,batchsize,hidden_size).to(device)
            #getting batchsize amount of encoder data
            input_tensor = getBatchData(train_encoder_input_data,batchsize,x).to(device)
            enc_hidden =temp
            enc_cell = temp
            if(input_tensor.size()[0] < batchsize):
                break
            #calling encoderforward
            output,(hidden,cell) = encoder.forward(input_tensor,enc_hidden,enc_cell)
            # to get batchsize amount of decoder input data for first hidden state
            input = train_decoder_input_data[x:x+batchsize,0].to(device).reshape(batchsize,1)
            #doing the resizeing for bidirectional implementation
            hidden = getResize(hidden,2,no_of_layers,batchsize,hidden_size)
            cell = getResize(cell,2,no_of_layers,batchsize,hidden_size)
            hiddenj = hidden
            #using teacher forcing
            teacher_forcing = True if random.random() < teacher_ratio else False
            k=0
            use_teacher_forcing = True if random.random() < teacher_ratio else False
            if teacher_forcing:
              #runing while loop for max no of characters in decoder data
                while k<max_dec_len:
                    output1,(hidden1,cell1) = decoder.forward(input,hidden,cell,hiddenj) # calling decoder forward
                    output2 = decoder.softmax(output1)
                    output3 = torch.argmax(output2,dim = 2)
                    y_pred.append(output1) # calling decoder forward
                    y_val.append(output3) # storing decoder predicted value index
                    # giving groung truth data for next input
                    input = train_decoder_input_data[x:x+batchsize,i].to(device).reshape(batchsize,1)
                    k+=1
            else:
                while k<max_dec_len:
                    output1,(hidden,cell) = decoder.forward(input,hidden,cell,hiddenj)  # calling decoder forward
                    output2 = decoder.softmax(output1)
                    output3 = torch.argmax(output2,dim = 2)
                    y_pred.append(output1)
                    y_val.append(output3)
                    # giving last output as next input
                    input = output3
                    k+=1
            #flattening the tensor 
            y_pred = getConcatenation(y_pred).to(device).reshape(max_dec_len*batchsize,len(target_token_index))
            y_val = getConcatenation(y_val)
            var=getBatchData(train_decoder_input_data,batchsize,x)
             #calling accuracy function to get accuracy
            total_acc += getAccuracy(var.to(device),y_val)
            #calculating loss
            loss = nn.CrossEntropyLoss(reduction = 'sum')(y_pred,var.reshape(-1).to(device))
            with torch.no_grad():
                total_loss += loss.item()
            loss.backward(retain_graph = True)
            #clipping the gradient
            torch.nn.utils.clip_grad_norm_(encoder.parameters(),max_norm = 1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(),max_norm = 1)
            opt_encoder.step()
            opt_decoder.step()
            opt_encoder.zero_grad()
            opt_decoder.zero_grad()
        denom=(len(train_decoder_input_data)*max_dec_len)
        training_loss = total_loss/denom # calculating training loss
        #calculating training accuracy
        training_accuracy = (total_acc/len(train_decoder_input_data))*100
        #calling getValidation to get validation loss and accuracy
        validation_loss,validation_accuracy = valevaluate(False,val_encoder_input_data,val_decoder_input_data,encoder,decoder,batchsize,hidden_size,embedding_size,no_of_layers)
        #test_loss,test_accuracy = Evaluate(False,test_encoder_input_data,test_decoder_input_data,encoder,decoder,batchsize,hidden_size,embedding_size,no_of_layers)        
        print('  loss = ',training_loss,'  accuracy = ',training_accuracy,'   validation loss= ',validation_loss,'  validation accuaracy= ',validation_accuracy)
        total_loss=0
        total_acc=0
    return encoder,decoder
def train(rnn,batchsize,hidden_size,embedding_size,no_of_layers,dropout,epochs):
    teacher_ratio = 0.5
    total_loss=0
    total_acc=0
    learning_rate=0.001
    #creating encoder and decoder model
    encoder = Encoder(rnn,embedding_size,hidden_size,no_of_layers,dropout).to(device)
    decoder = Decoder(rnn,embedding_size,hidden_size,no_of_layers,dropout,batchsize).to(device)
    opt_encoder = optim.Adam(encoder.parameters(),learning_rate)
    opt_decoder  = optim.Adam(decoder.parameters(),learning_rate)
    loss=0
    #running for loop epochs no of times
    for i in range(epochs):
        print('Epoch---',i+1,end=" ")
        #getting batchsize amount of encoder data
        for x in range(0,len(train_encoder_input_data),batchsize):
            y_pred = [] # to store predicted value by decoder to calculate loss
            y_val = []  #to store index of predicted value by decoder to calculate accuracy
            hidden_input = torch.zeros(no_of_layers,batchsize,hidden_size).to(device) # initial input to first hidden state(S0)
            temp=torch.zeros(2*no_of_layers,batchsize,hidden_size).to(device)
            input_tensor = getBatchData(train_encoder_input_data,batchsize,x).to(device) # to get batchsize of train data for encoder
            input_tensor_size=input_tensor.size()[0]
            #initializing with zeroes
            enc_hidden =temp
            enc_cell = temp
            if(batchsize>input_tensor_size):
              break
            #calling forward function of encoder
            output,(hidden,cell) = encoder.forward(input_tensor,enc_hidden,enc_cell)
            input = train_decoder_input_data[x:x+batchsize,0].to(device).reshape(batchsize,1) # to get batchsize amount of decoder input data for first hidden state
            #doing the resizeing for bidirectional implementation
            hidden = doResize(hidden,2,no_of_layers,batchsize,hidden_size)
            cell = doResize(cell,2,no_of_layers,batchsize,hidden_size)
            hiddenj = hidden
            #using teacher forcing 
            teacher_forcing = True if random.random() < teacher_ratio else False
            k=0
            if teacher_forcing:
                #runing while loop for max no of characters in decoder data
                while k<max_dec_len:
                    output1,(hidden,cell) = decoder.forward(input,hidden,cell,hiddenj) # calling decoder forward
                    output2 = decoder.softmax(output1)
                    output3 = torch.argmax(output2,dim = 2)
                    y_pred.append(output1) # storing decoder predicted value
                    y_val.append(output3) # storing decoder predicted value index
                    input = train_decoder_input_data[x:x+batchsize,i].to(device).reshape(batchsize,1) # giving groung truth data for next input
                    k+=1
            else:
                while k<max_dec_len:
                    output1,(hidden,cell) = decoder.forward(input,hidden,cell,hiddenj)  # calling decoder forward
                    output2 = decoder.softmax(output1)
                    output3 = torch.argmax(output2,dim = 2)
                    y_pred.append(output1)  # storing decoder predicted value
                    y_val.append(output3)   # storing decoder predicted value index
                    input = output3  # giving last output as next input
                    k+=1
            #flattening the tensor     
            y_pred = getConcatenation(y_pred).to(device).reshape(max_dec_len*batchsize,len(target_token_index)) # 
            y_val = getConcatenation(y_val)
            var=getBatchData(train_decoder_input_data,batchsize,x)
            total_acc += getAccuracy(var.to(device),y_val) #calling accuracy function to get accuracy
            #calculating loss
            loss = nn.CrossEntropyLoss(reduction = 'sum')(y_pred,var.reshape(-1).to(device))
            with torch.no_grad():
              total_loss += loss.item()
            loss.backward(retain_graph = True)
            #clipping the gradient
            torch.nn.utils.clip_grad_norm_(encoder.parameters(),max_norm = 1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(),max_norm = 1)
            opt_encoder.step()
            opt_decoder.step()
            opt_encoder.zero_grad()
            opt_decoder.zero_grad()
            loss=0
        del(y_pred)
        del(y_val)
        del(input)
        del(output1)
        del(output2)
        del(output3)
        del(hidden)
        del(cell)
        del(hiddenj)
        del(output)
        denom=(len(train_decoder_input_data)*max_dec_len)
        training_loss = total_loss/denom # calculating training loss
        #calculating training accuracy
        training_accuracy = (total_acc/len(train_decoder_input_data))*100
        #calling getValidation to get validation loss and accuracy
        validation_loss,validation_accuracy = getValidation(False,val_encoder_input_data,val_decoder_input_data,encoder,decoder,batchsize,hidden_size,embedding_size,no_of_layers)
        print(' Train loss = ',training_loss,' Train accuracy = ',training_accuracy,'   validation loss= ',validation_loss,'  validation accuaracy= ',validation_accuracy)
        total_loss=0
        total_acc=0


no_of_layers = args.no_of_layers
dropout = args.dropout
epochs = args.epochs
rnn = args.cell_type
batchsize = args.batchsize
hidden_size = args.hidden_size
char_embed_size = args.embedding_size

wandb.login()
wandb.init(project= args.wandb_project,entity = args.wandb_entity)
if(args.attentionRequired == True):
    Encoder,Decoder = attentiontrain(rnn,batchsize, hidden_size, char_embed_size, no_of_layers, dropout, epochs)
else:
    Encoder,Decoder = train(rnn,batchsize, hidden_size, char_embed_size, no_of_layers, dropout, epochs)
