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

## Content
* ```fusion.py``` --- get predictions using fusion model based on both BERT and NgramAttention models.
* ```train_ngram_attention.py``` --- train and evaluate NgramAttention model;
* ```train_ddp_ngram_attention.py``` --- train NgramAttention model on gpu in a distributed setup. Recommended, since training process is quite time consuming; 
* ```train_bert.py``` --- train and evaluate BERT model; 


## Data Format 
Training and evaluation data should be presented as a csv file with two columns: ```comment, label```. An example dataset can be found at ```hate_speech/out_data``` directory. Training data have to contain "train" in its name, data for evaluation have to contain "test" in its name.

To obtain predictions with a fusion model one can pass data in the same foram either with or without ```label``` column.

## Usage
#### Get prediction using fusion model:

 ```shell
python -m hate_speech.fusion --data_path <path to data> --save <path to csv file where predictions will be stored> 
```
Data should be presented in csv format with sentances to be assessed stored in a 'comment' column. If labels are also presented (in a 'label' column), F1-score and accuracy will be printed.

#### By default, model weights from the library will be used for making a prediction, but you can specify our own checkpoints for the fusion model via ```bert-ckpt-path``` and ```ngram-ckpt-path``` parameters:

```shell
python -m hate_speech.fusion --data_path <path to data> --save <path to csv file where predictions will be stored> --bert-ckpt-path <path to BERT ckpt> --ngram-ckpt-path <path to ngram-attention ckpt>
```

You can train and evaluate your own NgramAttention model as well as BERT using the following commands:

#### Train your NgramAttention model on cpu:
 ```shell
python -m hate_speech.train_ngram_attention --mode train --data_path <path to training set>
```

#### Train your NgramAttention model on gpu:
 ```shell
python -m hate_speech.train_ddp_ngram_attention --data_path <path to training set>
```
Learning rate, batch size, and number of epochs can be specified via options ```--learning_rate, --batch_size, --total_epochs```. All the script parameters can be found in ```train_ngram_attention.py``` or ```train_ddp_ngram_attention.py``` in case of distributed training.

#### Evaluate your NgramAttention model:
 ```shell
python -m hate_speech.train_ngram_attention --mode test --data_path <path to evaluation data>
```
To evaluate the model trained by you, specify the model checkpoint via ```--ckpt_path``` parameter.

#### Train and evaluate your BERT model:
 ```shell
python -m hate_speech.train_bert -- train-data-path <path to training data> --test-data-path <path to evaluation data> --save_ckpt <path, where to store final checkpoint>
```

