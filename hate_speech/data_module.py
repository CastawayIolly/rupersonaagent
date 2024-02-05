import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from torch import cuda
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from string import punctuation
import nltk
from pymystem3 import Mystem
from nltk.tokenize import TweetTokenizer 
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import fasttext
import fasttext.util
from torch import nn 

punctuations = list(punctuation) + ['-'] + ['...']
lem = Mystem()
#stop_words = set(stopwords.words('russian'))
tk = TweetTokenizer()

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment
        self.targets = self.data.label
        self.max_len = max_len
        self.eof = nn.Parameter(torch.randn(300), requires_grad=True)

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())
        if self.tokenizer is not None:
            inputs = self.tokenizer.encode_plus(
                comment_text,
                None,
                add_special_tokens=True,
                max_length=self.max_len,
                pad_to_max_length=True,
                #padding='max_length',
                return_token_type_ids=True
            )
            ids = inputs['input_ids']
            mask = inputs['attention_mask']
            token_type_ids = inputs["token_type_ids"]


            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(self.targets[index], dtype=torch.float)
            }
        
        else:
            tokens = tk.tokenize(comment_text)
            words_without_punkt = [i for i in tokens if ( i not in punctuations )]
            low_words = [i.lower() for i in words_without_punkt]
            if len(low_words) < 75:
                low_words += ['<EOS>']*(75 - len(low_words))  
            else:
                low_words = low_words[:75] 
            return {'lem_words': low_words, 
                    'targets': torch.tensor(self.targets[index], dtype=torch.float),
                    }    