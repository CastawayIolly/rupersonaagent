import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from torch import cuda
from tqdm import tqdm
import torch
import fasttext
from data_module import CustomDataset
import nltk

MAX_LEN = 75
TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 1
EPOCHS = 1
LEARNING_RATE = 0.001
tokenizer = None

fasttext.util.download_model('ru', if_exists='ignore') 
ft = fasttext.load_model('cc.ru.300.bin')

class NGramAttention(nn.Module):
    def __init__(self):
        super(NGramAttention, self).__init__()
        self.conv1 = nn.Conv1d(1, 256, 1)
        self.conv2 = nn.Conv1d(1, 256, 2)
        self.conv3 = nn.Conv1d(1, 256, 3)
        self.gru1 = nn.GRU(75, 75, dropout=0.2, bidirectional=True)
        self.gru2 = nn.GRU(74, 74, dropout=0.2, bidirectional=True)
        self.gru3 = nn.GRU(73, 73, dropout=0.2, bidirectional=True)
        self.attention1 = nn.MultiheadAttention(75, 1)
        self.attention2 = nn.MultiheadAttention(74, 1)
        self.attention3 = nn.MultiheadAttention(73, 1)
        self.dense1 = nn.Linear(222, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(128, 2)
        self.softmax = nn.Softmax()

        self.block1 = nn.Sequential(self.conv1, self.gru1, self.attention1)
        self.block2 = nn.Sequential(self.conv2, self.gru2, self.attention3)
        self.block3 = nn.Sequential(self.conv3, self.gru3, self.attention3)
        self.final_block = nn.Sequential(self.dense1,
                                        self.relu,
                                        self.dropout,
                                        self.dense2)  

    def forward(self, inputs):
        x = [
            self.block1(inputs),
            self.block2(inputs),
            self.block3(inputs),
            ]
        x = torch.cat(x, 2)
        out = self.final_block(x)
        return out
    

def main():
    device = 'cuda' if cuda.is_available() else 'cpu'
    
    # Creating the dataset and dataloader for the neural network
    df = pd.read_csv("out_data/ToxicRussianComments.csv")
     
    train_size = 0.8
    train_dataset=df.sample(frac=train_size,random_state=200)
    test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)
    
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    model = NGramAttention()
    model.to(device)


    def loss_fn(outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)
    
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    
    def train(epoch):
        model.train()
        for _,data in enumerate(tqdm(training_loader), 0):
            sentances = data['lem_words']
            targets = data['targets'].to(device, dtype = torch.float)
            embs = []
            for sentance in sentances: 
                s_embs = []
                for word in sentance:
                    if word != '<EOS>':
                       s_embs.append(ft.get_word_vector(word))
                    else:
                       s_embs.append(training_set.eof)    
                embs.append(s_embs)
            embs = torch.tensor(embs, dtype=torch.float)    
            outputs = model(embs)
            outputs = outputs.reshape(TRAIN_BATCH_SIZE)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            if _%50==0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for epoch in range(EPOCHS):
        train(epoch)        

if __name__ == "__main__":
    main()

