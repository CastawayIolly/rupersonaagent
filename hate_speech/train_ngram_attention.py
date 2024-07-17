import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch import cuda
from tqdm import tqdm
import torch
from data_module import CustomDataset
from sklearn.metrics import f1_score, accuracy_score
import os.path
import pickle
from ngram_attention_model import NGramAttention


# Config section
BATCH_SIZE = 5
EPOCHS = 5
LEARNING_RATE = 0.0001


def main(mode='train', total_epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
    device = 'cuda' if cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    # Create dataset for the model
    df = pd.read_csv("hate_speech/out_data/ToxicRussianComments.csv")

    train_size = 0.8
    train_dataset = df.sample(frac=train_size, random_state=200)
    test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    # Compute and save FastText embeddings for train dataset, if not already precomputed
    if not os.path.isfile('hate_speech/fasttext_train.pkl'):
        training_set = CustomDataset(train_dataset, regime='train')
        training_set.save_ft_feats()
    else:
        with open('hate_speech/fasttext_train.pkl', 'rb') as fp:
            training_set = pickle.load(fp) 
    training_set = TensorDataset(training_set[0], training_set[1])
    print('FastText embeddings for train set loaded.')

    # Compute and save FastText embeddings for test dataset, if not already precomputed
    if not os.path.isfile('hate_speech/fasttext_test.pkl'):
        testing_set = CustomDataset(test_dataset, regime='test')
        testing_set.save_ft_feats()
    else:
        with open('hate_speech/fasttext_test.pkl', 'rb') as fp:
            testing_set = pickle.load(fp)
    testing_set = TensorDataset(testing_set[0], testing_set[1])
    print('FastText embeddings for text set loaded.')

    train_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'drop_last': True,
                    'num_workers': 0}

    test_params = {'batch_size': batch_size,
                   'shuffle': True,
                   'drop_last': True,
                   'num_workers': 0}

    # Create dataloaders
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    # Initialize model 
    model = NGramAttention()
    model.to(device)

    # Define loss function 
    def loss_fn(outputs, targets):
        return torch.nn.CrossEntropyLoss(torch.tensor([0.4, 0.6], dtype=torch.float).to(device='cuda'))(outputs, targets)

    # Initialize optimizer and scheduler for further training
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    def train(epoch):
        model.train()
        for index, data in enumerate(tqdm(training_loader), 0):
            sentences = data[0]
            targets_ = data[1]
            targets = torch.empty((len(data[1]), 2), dtype=torch.float)
            # Substitute 0 in labels with [1., 0.] and 1 in labels with [0., 1.]
            for i, tar in enumerate(targets_):
                if tar == 0:
                    targets[i] = torch.tensor([1., 0.])
                else:
                    targets[i] = torch.tensor([0., 1.])
            # Preprocessing
            for idx, sentence in enumerate(sentences):
                for i, word in enumerate(sentence):
                    # If emb is pure zeros, then it alter it into trainable eos embedding
                    if torch.all(word.eq(torch.zeros_like(word))):
                        with torch.no_grad():
                            sentences[idx][i] = model.eos
            sentences = torch.unsqueeze(sentences, 1)

            # Training loop content
            outputs = model(sentences.to(device, dtype=torch.float))
            optimizer.zero_grad()
            loss = loss_fn(outputs, targets.to(device, dtype=torch.float))
            # Log loss every 50 iterations
            if index % 50 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            loss.backward()
            optimizer.step()
        scheduler.step()
        # Log learning rate when epoch ends
        print("Current LR: ", scheduler.get_last_lr())
        # Save checkpoint after every epoch
        ckpt = model.state_dict()
        if not os.path.isdir("hate_speech/ngram_checkpoints"):
            os.mkdir("hate_speech/ngram_checkpoints")
        ckpt_path = f"hate_speech/ngram_checkpoints/checkpoint_{epoch}.pt"
        torch.save(ckpt, ckpt_path)
        print(f"Epoch {epoch} | Training checkpoint saved at {ckpt_path}")

    def test(checkpoint_num=total_epochs-1):
        print("Loading model from checkpoint...")
        checkpoint = torch.load(f'ngram_checkpoints/checkpoint_{checkpoint_num}.pt')
        model.load_state_dict(checkpoint)
        print("Start evaluating...")
        results = []
        ans = []
        model.eval()
        for _, data in enumerate(tqdm(testing_loader), 0):
            sentences = data[0]
            targets = data[1]
            # Preprocessing
            for idx, sentence in enumerate(sentences):
                for i, word in enumerate(sentence):
                    # If emb is pure zeros, then alter it into trainable eos embedding
                    if torch.all(word.eq(torch.zeros_like(word))):
                        with torch.no_grad():
                            sentences[idx][i] = model.eos
            sentences = torch.unsqueeze(sentences, 1)

            # Evaluation loop content
            outputs = model(sentences.to(device, dtype=torch.float))
            results += outputs
            ans += targets

        # Compute and log metrics
        results = np.array([r.cpu().numpy() for r in results])
        results = [(1 if result[1] > result[0] else 0) for result in results]
        ans = np.array([r.cpu().numpy() for r in ans])
        f1 = f1_score(results, ans, average='weighted')
        acc = accuracy_score(results, ans)
        print(f'len test: {len(results)}\n F1: {f1}\n Accuracy: {acc}\n')
        print("Evaluation ended.")

    if mode == 'train':
        for epoch in range(total_epochs):
            train(epoch)
    if mode == 'test':
        test()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, help='Wheter to run script for model training or for test (default: train)')
    parser.add_argument('--total_epochs', default=EPOCHS, type=int, help='Total epochs to train the model (default: EPOCHS from config section)')
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='Input batch size (default: BATCH_SIZE from config section)')
    parser.add_argument('--learning_rate', default=LEARNING_RATE, type=float, help='learning rate for train mode (default: LEARNING_RATE from config section)')
    args = parser.parse_args()
    main(mode=args.mode)
