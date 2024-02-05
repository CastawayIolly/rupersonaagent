import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from torch import cuda
from tqdm import tqdm
from data_module import CustomDataset

# Sections of config

# Defining some key variables that will be used later on in the training

# proxy=http://proxy.ad.speechpro.com:3128
# export HTTP_PROXY=$proxy
# export http_proxy=$proxy
# export HTTPS_PROXY=$proxy
# export https_proxy=$proxy
# export ftp_proxy=$proxy
# export FTP_PROXY=$proxy
# export ALL_PROXY=$proxy
# export NO_PROXY=".stc,localhost,ad.speechpro.com" 
# export no_proxy=$NO_PROXY

MAX_LEN = 500
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 1)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output    
    
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

    model = BERTClass()
    model.to(device)

    def loss_fn(outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)
    
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
    
    def train(epoch):
        model.train()
        for _,data in enumerate(tqdm(training_loader), 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids)
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


