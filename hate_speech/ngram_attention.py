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


# config section
MAX_LEN = 75
TRAIN_BATCH_SIZE = 10
VALID_BATCH_SIZE = 10
EPOCHS = 5
LEARNING_RATE = 0.0001
tokenizer = None
EMBED_LEN = 300
MODE = 'train'

fasttext.util.download_model('ru', if_exists='ignore')
ft = fasttext.load_model('cc.ru.300.bin')

class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims
        
    def forward(self, x):
        x = x.permute(self.dims).contiguous()
        return x

class NGramAttention(nn.Module):
    def __init__(self):
        super(NGramAttention, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, (1, 1))
        self.conv2 = nn.Conv2d(1, 256, (2, 1))
        self.conv3 = nn.Conv2d(1, 256, (3, 1))

        self.perm1 = Permute((1, 3, 0, 2))
        self.perm2 = Permute((1, 3, 0, 2))
        self.perm3 = Permute((1, 3, 0, 2))

        self.flatten1 = nn.Flatten(0, 1)
        self.flatten2 = nn.Flatten(0, 1)
        self.flatten3 = nn.Flatten(0, 1)

        self.gru1 = nn.GRU(75, 16, num_layers=3, dropout=0.2, bidirectional=True)
        self.gru2 = nn.GRU(74, 16, num_layers=2, dropout=0.2, bidirectional=True)
        self.gru3 = nn.GRU(73, 16, num_layers=1, dropout=0.2, bidirectional=True)

        self.lin_for_att1 = nn.Linear(32, 75)
        self.lin_for_att2 = nn.Linear(32, 74)
        self.lin_for_att3 = nn.Linear(32, 73)

        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.tanh3 = nn.Tanh()

        self.ngram_context1 = nn.Parameter(torch.randn(75), requires_grad=True)
        self.ngram_context2 = nn.Parameter(torch.randn(74), requires_grad=True)
        self.ngram_context3 = nn.Parameter(torch.randn(73), requires_grad=True)

        self.soft1 = nn.Softmax()
        self.soft2 = nn.Softmax()
        self.soft3 = nn.Softmax()

        self.dense1 = nn.Linear(222, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(128, 2)

        self.block1 = nn.Sequential(self.conv1, self.perm1, self.flatten1, self.gru1)
        self.block2 = nn.Sequential(self.conv2, self.perm2, self.flatten2, self.gru2)
        self.block3 = nn.Sequential(self.conv3, self.perm3, self.flatten3, self.gru3)
        self.final_block = nn.Sequential(self.dense1,
                                    self.relu,
                                    self.dropout,
                                    self.dense2)
        self.softmax = nn.Softmax(dim=1)
        self.eos = nn.Parameter(torch.randn(300), requires_grad=True)

    def forward(self, inputs):
        x1, _ = self.block1(inputs)
        x1 = self.tanh1(self.lin_for_att1(x1))
        alpha1 = self.soft1(torch.matmul(x1, self.ngram_context1))
        x1 = torch.sum((x1.transpose(0, 2).transpose(1, 2)*alpha1), axis=1)

        x2, _ = self.block2(inputs)
        x2 = self.tanh2(self.lin_for_att2(x2))
        alpha2 = self.soft2(torch.matmul(x2, self.ngram_context2))
        x2 = torch.sum((x2.transpose(0, 2).transpose(1, 2)*alpha2), axis=1)

        x3, _ = self.block3(inputs)
        x3 = self.tanh3(self.lin_for_att3(x3))
        alpha3 = self.soft3(torch.matmul(x3, self.ngram_context3))
        x3 = torch.sum((x3.transpose(0, 2).transpose(1, 2)*alpha3), axis=1)

        x = torch.cat([x1, x2, x3], dim = 0)
        # print(x)
        out = self.final_block(x.transpose(0, 1))
        # print(f'out: {out.shape}')
        return out # self.softmax(out)


def main():
    device = 'cuda' if cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    # Creating the dataset and dataloader for the neural network
    df = pd.read_csv("out_data/ToxicRussianComments.csv")

    train_size = 0.8
    train_dataset=df.sample(frac=train_size, random_state=200)
    test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    if not os.path.isfile('fasttext_train.pkl'):
        training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN, 'train')
        training_set.save_ft_feats()
    else:
        with open('fasttext_train.pkl', 'rb') as fp:
            training_set = pickle.load(fp)
    training_set = TensorDataset(training_set[0], training_set[1]) 

    if not os.path.isfile('fasttext_test.pkl'):
        testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN, 'test')
        testing_set.save_ft_feats()
    else:
        with open('fasttext_test.pkl', 'rb') as fp:
            testing_set = pickle.load(fp)
    testing_set = TensorDataset(testing_set[0], testing_set[1])

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'drop_last': True,
                'num_workers': 0
                }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'drop_last': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    model = NGramAttention()
    model.to(device)


    def loss_fn(outputs, targets):
        return torch.nn.CrossEntropyLoss(torch.tensor([0.4, 0.6], dtype=torch.float).to(device='cuda'))(outputs, targets)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    def train(epoch):
        model.train()
        for index, data in enumerate(tqdm(training_loader), 0):
            # preprocessing
            sentences = data[0]
            targets_ = data[1] # .to(device, dtype = torch.float)
            targets = torch.empty((len(data[1]), 2), dtype=torch.float)
            for i, tar in enumerate(targets_):
                if tar == 0:
                    targets[i] = torch.tensor([1.,0.])
                else:
                    targets[i] = torch.tensor([0.,1.])

            for idx, sentence in enumerate(sentences):
                for i, word in enumerate(sentence):
                    # if emb is pure zeros, then it is altered into trainable eos embedding
                    if torch.all(word.eq(torch.zeros_like(word))):
                        with torch.no_grad():
                            sentences[idx][i] = model.eos
            sentences = torch.unsqueeze(sentences, 1)

            with torch.enable_grad():
                outputs = model(sentences.to(device, dtype=torch.float))
                # outputs = outputs.reshape(TRAIN_BATCH_SIZE)

                optimizer.zero_grad()
                loss = loss_fn(outputs, targets.to(device, dtype=torch.float))
                if index %50 ==0:
                    print(f'Epoch: {epoch}, Loss:  {loss.item()}')

                loss.backward()
                optimizer.step()
        scheduler.step()
        print("Current LR: ", scheduler.get_last_lr())
        ckp = model.state_dict()
        PATH = f"/mnt/cs/voice/korenevskaya-a/nirma/checkpoints_ngram_attention/checkpoint_{epoch}.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def test():
        print("Start evaluating...")
        results = []
        ans = []
        model.eval()
        for _,data in enumerate(tqdm(testing_loader), 0):
            # preprocessing
            sentences = data[0]
            targets = data[1] # .to(device, dtype = torch.float)
            # for i, tar in enumerate(targets_):
            #     if tar == 0:
            #         targets[i] = torch.tensor([1.,0.])
            #     else:
            #         targets[i] = torch.tensor([0.,1.])

            for idx, sentence in enumerate(sentences):
                for i, word in enumerate(sentence):
                    # if emb is pure zeros, then it is altered into trainable eos embedding
                    if torch.all(word.eq(torch.zeros_like(word))):
                        with torch.no_grad():
                            sentences[idx][i] = model.eos
            sentences = torch.unsqueeze(sentences, 1)

            with torch.no_grad():
                outputs = model(sentences.to(device, dtype=torch.float))
                results += outputs
                ans += targets

        results = np.array([r.cpu().numpy() for r in results])
        results = [(1 if result[1] > result[0] else 0) for result in results]
        ans = np.array([r.cpu().numpy() for r in ans])
        f1 = f1_score(results, ans, average='weighted')
        acc = accuracy_score(results, ans)
        print(f'len test: {len(results)}\n F1: {f1}\n Accuracy: {acc}\n')
        print("Evaluation ended.")


    def check():
        print("Start checking...")
        results = []
        ans = []
        model.eval()
        for idx ,data in enumerate(tqdm(testing_loader), 0):
            if idx < 40:
                # preprocessing
                sentences = data[0]
                targets = data[1] # .to(device, dtype = torch.float)
                # for i, tar in enumerate(targets_):
                #     if tar == 0:
                #         targets[i] = torch.tensor([1.,0.])
                #     else:
                #         targets[i] = torch.tensor([0.,1.])

                for idx, sentence in enumerate(sentences):
                    for i, word in enumerate(sentence):
                        # if emb is pure zeros, then it is altered into trainable eos embedding
                        if torch.all(word.eq(torch.zeros_like(word))):
                            with torch.no_grad():
                                sentences[idx][i] = model.eos
                sentences = torch.unsqueeze(sentences, 1) 

                with torch.no_grad():
                    outputs = model(sentences.to(device, dtype=torch.float))
                    results += outputs
                    ans += targets

        results = np.array([r.cpu().numpy() for r in results])
        print(results)
        results = [(1 if result[1] > result[0] else 0) for result in results]
        print(results)
        ans = np.array([r.cpu().numpy() for r in ans])
        print(f"ans sum: {np.sum(ans)}")
        print(f"ans:{ans}")
        f1 = f1_score(results, ans, average='weighted')
        acc = accuracy_score(results, ans)
        print(f'len test: {len(results)}\n F1: {f1}\n Accuracy: {acc}\n')

    if MODE == 'train':
        for epoch in range(EPOCHS):
            train(epoch)
    if MODE == 'test':
        checkpoint = torch.load('checkpoints_ngram_attention/checkpoint_4.pt')
        model.load_state_dict(checkpoint)
        test()
    if MODE == 'check':
        checkpoint = torch.load('checkpoints_ngram_attention/checkpoint_4.pt')
        model.load_state_dict(checkpoint)
        check()


if __name__ == "__main__":
    main()
