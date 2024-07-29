import torch
from torch.utils.data import Dataset, TensorDataset
from string import punctuation
from pymystem3 import Mystem
from nltk.tokenize import TweetTokenizer
import fasttext.util
from torch import nn
import pandas as pd


fasttext.util.download_model('ru', if_exists='ignore')
ft = fasttext.load_model('cc.ru.300.bin')

punctuations = list(punctuation) + ['-'] + ['...']
lem = Mystem()
tk = TweetTokenizer()


def set_ngram_dataset(data_path):
    # Create dataset for the model
    if isinstance(data_path, str):
        dataset = pd.read_csv(data_path)
    else:
        dataset = data_path

    if 'train' in data_path:
        dataset_type = 'train '
    elif 'test' in data_path:
        dataset_type = 'test '
    else:
        dataset_type = ''
    print(f"{dataset_type}Dataset: {dataset.shape}")
    dataset = CustomDataset(dataset, regime=dataset_type)

    data_feats = torch.zeros((len(dataset), dataset[0]['features'].shape[0], dataset[0]['features'].shape[1]))
    for i in range(len(dataset)):
        data_feats[i] = dataset[i]['features']
    dataset = TensorDataset(data_feats, dataset[:]['targets'])
    return dataset


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
