# CS6910-Assignment-3

## Question 1

1.The data is uploaded to colab and is read by the ipynb file.<br>
2.For the task of transliteration, I used the Hindi Language dataset.<br>
3.Read the train, valid, and test data using pandas.<br>
4.First, i did some data processing<br>
5.created Encoder, Decoder, and train functions.<br>
6.code is flexible so that cell,no_of_layers,input_embedding_size,batchsize,hidden_size can be changed and passed as parameters.<br>


## Question 2

1.Created Functions for Encoder,Decoder,getValidation(evaluates the validation data),vanilla function(used in sweeps) and sweep configuration

The sweep configuration is :
```python

sweep_configuration = {
    'method' : 'bayes',
    'metric' : { 'goal' : 'maximize',
    'name' : 'validation_accuracy'},
    'parameters':{
        'cell_type' : {'values' : ['LSTM','RNN','GRU']},
        'batchsize' : {'values' : [128,256]},
        'input_embedding_size' : {'values' : [128,256,512,1024]},
        'dropout' : {'values' : [0.1,0.2,0.3,0.4,0.5]},
        'no_of_layers' : {'values' : [2,3,4]},
        'hidden_size' : {'values' : [128,256,512,1024]},
        'bidirectional' : {'values' : ['Yes']},
        'epochs' : {'values' : [10,20,30]}
    }
}

```

### Steps to build the Seq2Seq Network.

1.The Functions inside the Q2 file are encoder,decoder and train that takes instances of encoder and decoder with specified parameters

An Instance of Encoder is as follows:

```python
 encoder = Encoder(cell,embedding_size,hidden_size,no_of_layers,dropout)
```

It can be implemented using the following parameters:

-cell = LSTM or GRU or RNN

- embedding_size = Embedding size required to get a representation of a character.

- hidden_size = Size of cell state of RNN,LSTM,GRU

- no_of_layers = no of hidden layers of RNN,LSTM,GRU .

- dropout = ranges between 0-1. denotes the probability to dropout.


An Instance of a Decoder is as follows:

``` python
 decoder = Decoder(cell,embedding_size,hidden_size,no_of_layers,dropout,batchsize)
```
Parameters are same as Encoder. Only difference is Encoder is bidirectional.
  
### Training the Seq2Seq Network


The model can be trained using the `train` function.

- for an instance of Seq2Seq network given earlier we can train it by calling the train function as follows:

```python
    Encoder,Decoder = train(cell,batchsize,hidden_size,embedding_size,no_of_layers,dropout,epochs)
```

## Question 4

Evaluated Accuracy for Test data after getting the best configuration from sweeps in Q2.ipynb

Best configuration is :
 ``` python
    batchsize = 256
    hidden_size = 1024
    embedding_size = 256
    no_of_layers = 3
    dropout = 0.4
    epochs = 20
    rnn = 'LSTM'
 ```
 - Stored the predictions using the Evaluate function :
 
 ``` python
    test_loss,test_accuracy,predictions = Evaluate(False,test_encoder_input_data,test_decoder_input_data,Encoder,Decoder,batchsize,hidden_size,embedding_size,no_of_layers)
 ```
 - Using this predictions which is a dataframe having columns as original and predicted hindi words.
 
 - Store this using to_excel method of pandas into a folder Prediction_vanilla

## Question 5:

Evaluated Accuracy for Test data after getting the best configuration from sweeps for Attention model.

Best Configuration is :

``` python
batchsize = 512
hidden_size = 1024
embedding_size = 1024
no_of_layers = 1
dropout = 0.4
epochs = 30
rnn = 'LSTM'
```

- Got the test_loss,Test_accuracy and predictions using the Evaluate function:

``` python
test_loss,test_accuracy,predictions = Evaluate(True,test_encoder_input_data,test_decoder_input_data,Encoder1,Decoder1,batchsize,hidden_size,embedding_size,no_of_layers)
```

- Using this predictions which is a dataframe having columns as original and predicted hindi words .

- Store this using to_excel method of pandas into a folder Prediction_vanilla.
  
## Instructions about train.py

- For train.py you need to login into your wandb account from the terminal using the api key.

- In the arguments into the terminal please give valid credentials like project_name and entity_name.

- List of argument supported by train.py is :

``` python
    -wp,--wandb_project = wandb project name
    -we,--wandb_entity = wandb entity name
    -e,--epochs = no of epochs
    -b,--batchsize = size of batch
    -hidden,--hidden_size = size of cell state
    -embed, -- embedding_size = embedding size
    -cell,--cell_type = cell type choices = ['LSTM','GRU','RNN']
    -drop,--dropout = probablity of dropout
    -attn,--attentionRequired = choices = [True,False]
    -layer,--no_of_layers = no of stack of RNN or LSTM or GRU
    
```

- For calling train.py with appropriate argument please follow the given example:

  `python train.py -wp projectname -we entity name -e 10 -b 128 -hidden 1024 -embed 1024 -cell 'LSTM' -drop 0.3 -attn True -layer 2 
  




  

  

  

  
  

 
