from torch import nn
import torch


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

        self.gru1 = nn.GRU(75, 32, num_layers=4, dropout=0.2, bidirectional=True)
        self.gru2 = nn.GRU(74, 32, num_layers=3, dropout=0.2, bidirectional=True)
        self.gru3 = nn.GRU(73, 32, num_layers=2, dropout=0.2, bidirectional=True)

        self.lin_for_att1 = nn.Linear(64, 75)
        self.lin_for_att2 = nn.Linear(64, 74)
        self.lin_for_att3 = nn.Linear(64, 73)

        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.tanh3 = nn.Tanh()

        self.ngram_context1 = nn.Parameter(torch.randn(75), requires_grad=True)
        self.ngram_context2 = nn.Parameter(torch.randn(74), requires_grad=True)
        self.ngram_context3 = nn.Parameter(torch.randn(73), requires_grad=True)

        self.soft1 = nn.Softmax(dim=0)
        self.soft2 = nn.Softmax(dim=0)
        self.soft3 = nn.Softmax(dim=0)

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
        x1 = torch.sum((x1.transpose(0, 2).transpose(1, 2) * alpha1), axis=1)

        x2, _ = self.block2(inputs)
        x2 = self.tanh2(self.lin_for_att2(x2))
        alpha2 = self.soft2(torch.matmul(x2, self.ngram_context2))
        x2 = torch.sum((x2.transpose(0, 2).transpose(1, 2) * alpha2), axis=1)

        x3, _ = self.block3(inputs)
        x3 = self.tanh3(self.lin_for_att3(x3))
        alpha3 = self.soft3(torch.matmul(x3, self.ngram_context3))
        x3 = torch.sum((x3.transpose(0, 2).transpose(1, 2) * alpha3), axis=1)

        x = torch.cat([x1, x2, x3], dim=0)
        out = self.final_block(x.transpose(0, 1))
        return out

    def preprocess(self, sentences):
        for idx, sentence in enumerate(sentences):
            for i, word in enumerate(sentence):
                # If emb is pure zeros, then it alter it into trainable eos embedding
                if torch.all(word.eq(torch.zeros_like(word))):
                    with torch.no_grad():
                        sentences[idx][i] = self.eos
        sentences = torch.unsqueeze(sentences, 1)
        return sentences
