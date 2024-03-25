import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import transformers
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from torch import cuda
from tqdm import tqdm
import torch
import fasttext
import nltk

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from data_module import CustomDataset
from ngram_attention import NGramAttention

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


MAX_LEN = 75
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001
tokenizer = None
EMBED_LEN = 300

fasttext.util.download_model('ru', if_exists='ignore') 
ft = fasttext.load_model('cc.ru.300.bin')

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
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
    
    def loss_fn(self, outputs, targets):
        return torch.nn.CrossEntropyLoss()(outputs, targets)


    def _run_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)
        
        for _,data in enumerate(tqdm(self.train_data), 0):
            sentences = data['lem_words']
            sentences = list(zip(*[list(x) for x in sentences]))
            targets = torch.tensor([[1,0] if tar==0 else [0,1] for tar in data['targets']], device=self.gpu_id, dtype=torch.float)
            #print(targets)
            embs = torch.empty((len(sentences), MAX_LEN, EMBED_LEN)).to(self.gpu_id, dtype = torch.float)
            for idx, sentence in enumerate(sentences): 
                s_embs =torch.empty((MAX_LEN, EMBED_LEN)).to(self.gpu_id, dtype = torch.float)
                for i, word in enumerate(sentence):
                    if word != '<EOS>':
                       s_embs[i] = torch.from_numpy(ft.get_word_vector(word))
                    else:
                       s_embs[i] = self.train_data.dataset.eof   
                embs[idx] = s_embs    
            embs = torch.unsqueeze(embs, 1)
            outputs = self.model(embs)
            
            self.optimizer.zero_grad()
            loss = self.loss_fn(outputs, targets)
            if _%50==0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def run_test(self,):
        results = [] #torch.empty((len(test_dataset)))
        ans = [] #torch.empty((len(test_dataset)))
        with torch.no_grad():
            for _,data in enumerate(tqdm(self.test_data), 0):
                sentences = data['lem_words']
                sentences = list(zip(*[list(x) for x in sentences]))
                targets = torch.tensor([[1,0] if tar==0 else [0,1] for tar in data['targets']], device=self.gpu_id, dtype=torch.float)
                print(targets)
                embs = torch.empty((len(sentences), MAX_LEN, EMBED_LEN)).to(self.gpu_id, dtype = torch.float)
                for idx, sentence in enumerate(sentences): 
                    #print(sentence)
                    s_embs =torch.empty((MAX_LEN, EMBED_LEN)).to(self.gpu_id, dtype = torch.float)
                    for i, word in enumerate(sentence):
                        if word != '<EOS>':
                            s_embs[i] = torch.from_numpy(ft.get_word_vector(word))
                        else:
                            s_embs[i] = self.train_data.dataset.eof   
                    embs[idx] = s_embs    
                embs = torch.unsqueeze(embs, 1)
                outputs = self.model(embs)
                outputs = outputs.squeeze()
                #print(outputs)    
                
                results += outputs
                ans += targets
        results = np.array([r.cpu().numpy() for r in results])
        results = [(1 if result[1] == 1 else 0) for result in results]
        ans = np.array([r.cpu().numpy() for r in ans])
        f1 = f1_score(results, ans, average='weighted')
        acc = accuracy_score(results, ans)
        print(f'len test: {len(results)}\n F1: {f1}\n Accuracy: {acc}\n')            
    
    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = f"/mnt/cs/voice/korenevskaya-a/nirma/checkpoints_ngram_attention/checkpoint_{epoch}.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            #self.run_test()
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    df = pd.read_csv("out_data/ToxicRussianComments.csv")
     
    train_size = 0.8
    train_dataset=df.sample(frac=train_size,random_state=200)
    test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    train_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    test_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

    model = NGramAttention()
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
    return train_set, test_set, model, optimizer


def prepare_dataloader(train_dataset: Dataset, test_dataset: Dataset, batch_size: int):
    train_data = DataLoader(train_dataset,
                            batch_size,
                            pin_memory=True,
                            num_workers = 1, 
                            shuffle=False,
                            drop_last=True,
                            sampler=DistributedSampler(train_dataset))
    test_data = DataLoader(test_dataset,
                            batch_size,
                            pin_memory=True,
                            num_workers = 1, 
                            shuffle=False,
                            drop_last=False,)
                            #sampler=DistributedSampler(dataset))
    return train_data, test_data


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    train_dataset, test_dataset, model, optimizer = load_train_objs()
    train_data, test_data = prepare_dataloader(train_dataset, test_dataset, batch_size)
    trainer = Trainer(model, train_data, test_data, optimizer, rank, save_every)
    #trainer.train(total_epochs)
    trainer.run_test()
    destroy_process_group()


if __name__ == "__main__":
    print("Cuda support:", torch.cuda.is_available(),":", torch.cuda.device_count(), "devices")
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=EPOCHS, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=1, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=TRAIN_BATCH_SIZE, type=int, help='Input batch size on each device (default: TRAIN_BATCH_SIZE)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)