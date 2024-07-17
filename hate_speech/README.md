# Hate speech detector

## Requirenments
* torch
* transformers  
* fasttext
* scickit-learn
* pandas
* numpy
* random
* nltk 
* pymystem3

In order to use this module one have to delete datasets.__init__ file. Otherwise it may create a conflict with datasets module in transformers package.

## Content
* ```train_ngram_attention.py``` --- train and test NgramAttention model;
* ```train_ddp_ngram_attention.py``` --- train NgramAttention model on gpu in a distributed setup. Recommended, since training process is quite time consuming; 
* ```train_bert.py``` --- train BERT model; 
* ```fusion.py``` --- fuse BERT with NgramAttention for better quality.

## Data Format 
Data should be presented as a csv file with two columns: ```comment, label```, and have to be stored in ```out_data``` directory. An example dataset can be found at the same directory. 

## Usage

 Train NgramAttention model on cpu:
 ```shell
python -m hate_speech.train_ngram_attention --mode train
```

Train NgramAttention model on gpu:
 ```shell
python -m hate_speech.train_ddp_ngram_attention
```
Learning rate, batch size, and number of epochs can be specified via options ```--learning_rate, --batch_size, --total_epochs```. All the script parameters can be found in ```train_ngram_attention.py``` or ```train_ddp_ngram_attention.py``` in case of distributed training.

Evaluate NgramAttention model:
 ```shell
python -m hate_speech.train_ngram_attention --mode test
```
Train and evaluate BERT model:

 ```shell
python -m hate_speech.train_bert --mode test
```
Train and evaluate fusion model:

 ```shell
python -m hate_speech.fusion
```
