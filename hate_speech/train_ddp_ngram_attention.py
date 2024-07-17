import pandas as pd
from torch.utils.data import TensorDataset, Dataset, DataLoader
from tqdm import tqdm
import torch
import fasttext
import pickle

from data_module import CustomDataset
from ngram_attention_model import NGramAttention

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

# Config section
BATCH_SIZE = 5
EPOCHS = 5
LEARNING_RATE = 0.0001


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1234"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)

    def loss_fn(self, outputs, targets):
        return torch.nn.CrossEntropyLoss()(outputs, targets)

    def _run_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)

        for index, data in enumerate(tqdm(self.train_data), 0):
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
                            sentences[idx][i] = self.model.module.eos
            sentences = torch.unsqueeze(sentences, 1)

            # Training loop content
            outputs = self.model(sentences.to(self.gpu_id, dtype=torch.float))
            self.optimizer.zero_grad()
            loss = self.loss_fn(outputs, targets.to(self.gpu_id, dtype=torch.float))
            # Log loss every 50 iterations
            if index % 50 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            loss.backward()
            self.optimizer.step()
        self.scheduler.step()
        # Log learning rate when epoch ends
        print("Current LR: ", self.scheduler.get_last_lr())

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        if not os.path.isdir("hate_speech/ngram_checkpoints"):
            os.mkdir("hate_speech/ngram_checkpoints")
        PATH = f"hate_speech/ngram_checkpoints/checkpoint_{epoch}.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs(learning_rate):
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
    
    # Initialize model, optimizer and scheduler
    model = NGramAttention()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    
    return training_set, testing_set, model, optimizer, scheduler

def prepare_dataloader(train_dataset: Dataset, test_dataset: Dataset, batch_size: int):
    train_data = DataLoader(train_dataset,
                            batch_size,
                            pin_memory=True,
                            num_workers=1,
                            shuffle=False,
                            drop_last=True,
                            sampler=DistributedSampler(train_dataset))
    test_data = DataLoader(test_dataset,
                           batch_size,
                           pin_memory=True,
                           num_workers=1,
                           shuffle=False,
                           drop_last=False,)
    return train_data, test_data


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, learning_rate: float):
    ddp_setup(rank, world_size)
    train_dataset, test_dataset, model, optimizer, scheduler = load_train_objs(learning_rate)
    train_data, test_data = prepare_dataloader(train_dataset, test_dataset, batch_size)
    trainer = Trainer(model, train_data, test_data, optimizer, scheduler, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    print("Cuda support:", torch.cuda.is_available(), ":", torch.cuda.device_count(), "devices")
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=EPOCHS, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=1, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='Input batch size on each device (default: BATCH_SIZE from config section)')
    parser.add_argument('--learning_rate', default=LEARNING_RATE, type=float, help='Input batch size on each device (default: LEARNING_RATE from config section)')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size, args.learning_rate), nprocs=world_size)
