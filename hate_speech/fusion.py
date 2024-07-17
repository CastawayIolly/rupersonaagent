import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch import cuda
from tqdm import tqdm
import torch
import fasttext
from data_module import CustomDataset
from sklearn.metrics import f1_score, accuracy_score
import os.path
import pickle
from transformers import AutoModel, BertTokenizer, BertForSequenceClassification
from ngram_attention_model import NGramAttention
import torch.nn.functional as F


def main():
    bert = BertForSequenceClassification.from_pretrained('cointegrated/rubert-tiny2', num_labels=2).to("cuda")
    bert_ckpt = torch.load('hate_speech/bert_ckpt.pt')
    tokenizer = BertTokenizer.from_pretrained('cointegrated/rubert-tiny2')
    bert.load_state_dict(bert_ckpt)

    ngram = NGramAttention()
    ngram_ckpt = torch.load('hate_speech/ngram_checkpoints/checkpoint_4.pt')
    ngram.load_state_dict(ngram_ckpt)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    # Creating the dataset and dataloader for BERT model 
    df = pd.read_csv("hate_speech/out_data/ToxicRussianComments.csv")

    train_size = 0.8
    train_dataset=df.sample(frac=train_size,random_state=200)
    test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    MAX_LEN = 267

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))


    # Dataloaders for ngram_attention model    
    with open('hate_speech/fasttext_test.pkl', 'rb') as fp:
        testing_set = pickle.load(fp)       
    testing_set = TensorDataset(testing_set[0], testing_set[1]) 


    test_params = {'batch_size': 10,
                    'shuffle': False,
                    'drop_last': True,
                    'num_workers': 0
                    }

    testing_loader = DataLoader(testing_set, **test_params)
    # Вычисление веротяностей для тестовой и обучающей базы

    if not os.path.isfile('hate_speech/logits_ngram_test.pkl'):
        ngram.to(device)
        ngram.eval()
        results_ngram_test = []
        for _, data in enumerate(tqdm(testing_loader), 0):
            sentences = data[0]
            targets = data[1]
            # Preprocessing
            for idx, sentence in enumerate(sentences):
                for i, word in enumerate(sentence):
                    # If emb is pure zeros, then alter it into trainable eos embedding
                    if torch.all(word.eq(torch.zeros_like(word))):
                        with torch.no_grad():
                            sentences[idx][i] = ngram.eos
            sentences = torch.unsqueeze(sentences, 1)
        
            with torch.no_grad():
                outputs = ngram(sentences.to(device, dtype=torch.float))
                results_ngram_test += outputs
        with open('hate_speech/logits_ngram_test.pkl', 'wb') as f:
            pickle.dump(results_ngram_test, f)   
    else:
        with open('hate_speech/logits_ngram_test.pkl', 'rb') as f:
            results_ngram_test = pickle.load(f)    

    if not os.path.isfile('hate_speech/logits_bert_test.pkl'):
        results_bert_test = []
        bert.to(device='cpu')
        bert.eval()
        for comment in tqdm(test_dataset['comment']):
            input_ids = torch.tensor(tokenizer.encode(comment), device='cpu').unsqueeze(0)
            outputs = bert(input_ids)
            results_bert_test.append(outputs.logits)
        with open('logits_bert_test.pkl', 'wb') as f:
            pickle.dump(results_bert_test, f)     
    else:
        with open('logits_bert_test.pkl', 'rb') as f:
            results_bert_test = pickle.load(f)     


    ans_test = torch.tensor(test_dataset['label'].values[:len(results_ngram_test)])

    alpha = 0.8
    preds = []
    for i, pair in enumerate(zip(results_bert_test, results_ngram_test)): 
        pair0 = torch.sigmoid(pair[0].to(device).squeeze(0))
        pair1 = torch.sigmoid(pair[1])
        pred = pair0*alpha + pair1*(1-alpha)    
        preds.append(int(pred.argmax()))
    f1 = f1_score(ans_test[:len(preds)], preds)
    acc = np.sum((ans_test[:len(preds)]==preds))/len(ans_test)
    print("F1:",  f1)
    print("Acc:",  acc)

if __name__ == '__main__':
    main()    