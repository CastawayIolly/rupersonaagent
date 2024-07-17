import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from string import punctuation
from pymystem3 import Mystem
from nltk.tokenize import TweetTokenizer
import fasttext
import fasttext.util
from torch import nn
import pickle

fasttext.util.download_model('ru', if_exists='ignore')
ft = fasttext.load_model('cc.ru.300.bin')

punctuations = list(punctuation) + ['-'] + ['...']
lem = Mystem()
tk = TweetTokenizer()


class CustomDataset(Dataset):

    def __init__(self, dataframe, regime=None, tokenizer=None, max_len=None):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment
        self.targets = self.data.label
        self.max_len = max_len
        self.eof = nn.Parameter(torch.randn(300), requires_grad=True)
        self.regime = regime

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())
        # For BERT model 
        # Tokenization
        if self.tokenizer is not None:
            inputs = self.tokenizer.encode_plus(
                comment_text,
                max_length=self.max_len,
                padding='max_length',
                truncation=True
            )

            item = {k: torch.tensor(v) for k, v in inputs.items()}
            item["labels"] = torch.tensor([self.targets[index]])
            return item

        # For ngram_attention model
        else:
            # Tokenization and preprocessing
            tokens = tk.tokenize(comment_text)
            words_without_punkt = [i for i in tokens if (i not in punctuations)]
            low_words = [i.lower() for i in words_without_punkt]

            # Compute fasttext features and pad for len 75 if necessary
            features = torch.empty((75, 300), dtype=torch.float)
            for i, word in enumerate(low_words):
                if i < 75:
                    features[i, :] = torch.from_numpy(ft.get_word_vector(word))
                else:
                    break
            for i in range(len(low_words), 75):
                features[i] = torch.zeros((300))

            return {'features': features,
                    'targets': torch.tensor(self.targets[index], dtype=torch.float),
                    }

    def save_ft_feats(self):
        # Compute fasttext embeddings for ngram_attention model and save into .pkl
        embeddings = torch.empty(((len(self.data)), 75, 300), dtype=torch.float)
        targets = torch.empty(((len(self.data)), 1), dtype=torch.float)
        print(f'Compute FastText embeddings for {self.regime} dataset...')
        
        for i in tqdm(range(len(self.data))):
            dict_ = self.__getitem__(i)
            embeddings[i] = (dict_['features'])
            targets[i] = (dict_['targets'])
        
        save = [embeddings, targets]
        print(f'Save {self.regime} embeddings into fasttext_{self.regime}.pkl')
        with open(f'fasttext_{self.regime}.pkl', 'wb') as f:
            pickle.dump(save, f)
        print('Success!')
