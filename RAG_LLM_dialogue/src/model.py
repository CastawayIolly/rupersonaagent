import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModel


class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=64):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        query = self.dataset[idx]['query']
        candidate = self.dataset[idx]['candidate']
        query_inputs = self.tokenizer(
            query,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        candidate_inputs = self.tokenizer(
            candidate,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'query_input_ids': query_inputs['input_ids'].squeeze(0),
            'query_attention_mask': query_inputs['attention_mask'].squeeze(0),
            'candidate_input_ids': candidate_inputs['input_ids'].squeeze(0),
            'candidate_attention_mask': candidate_inputs['attention_mask'].squeeze(0)
        }


class BiEncoder(pl.LightningModule):
    def __init__(self, model_name='cointegrated/rubert-tiny2', lr=2e-5):
        super(BiEncoder, self).__init__()
        self.query_encoder = AutoModel.from_pretrained(model_name)
        self.candidate_encoder = AutoModel.from_pretrained(model_name)
        self.loss_fn = nn.CosineEmbeddingLoss(margin=0.5)
        self.lr = lr

    def forward(self, query_input_ids, query_attention_mask, candidate_input_ids, candidate_attention_mask):
        query_outputs = self.query_encoder(query_input_ids, query_attention_mask)
        candidate_outputs = self.candidate_encoder(candidate_input_ids, candidate_attention_mask)
        query_embeddings = query_outputs.last_hidden_state[:, 0, :]
        candidate_embeddings = candidate_outputs.last_hidden_state[:, 0, :]
        return query_embeddings, candidate_embeddings

    def training_step(self, batch, batch_idx):
        query_input_ids = batch['query_input_ids']
        query_attention_mask = batch['query_attention_mask']
        candidate_input_ids = batch['candidate_input_ids']
        candidate_attention_mask = batch['candidate_attention_mask']
        query_embeddings, candidate_embeddings = self(
            query_input_ids, query_attention_mask, candidate_input_ids, candidate_attention_mask
        )
        labels = torch.ones(query_embeddings.size(0), device=self.device)
        loss = self.loss_fn(query_embeddings, candidate_embeddings, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        query_input_ids = batch['query_input_ids']
        query_attention_mask = batch['query_attention_mask']
        candidate_input_ids = batch['candidate_input_ids']
        candidate_attention_mask = batch['candidate_attention_mask']
        query_embeddings, candidate_embeddings = self(
            query_input_ids, query_attention_mask, candidate_input_ids, candidate_attention_mask
        )
        labels = torch.ones(query_embeddings.size(0), device=self.device)
        loss = self.loss_fn(query_embeddings, candidate_embeddings, labels)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
