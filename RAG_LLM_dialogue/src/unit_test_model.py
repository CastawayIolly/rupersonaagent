import torch
from hamcrest import assert_that, equal_to
from model import BiEncoder, CustomDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets import load_from_disk


class TestBiEncoderModel:
    @classmethod
    def setup_class(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
        dataset_path = 'data/toloka_data'
        dataset = load_from_disk(dataset_path)
        cls.train_dataset = CustomDataset(dataset['train'], cls.tokenizer, max_length=64)
        cls.val_dataset = CustomDataset(dataset['val'], cls.tokenizer, max_length=64)
        cls.train_dataloader = DataLoader(cls.train_dataset, batch_size=8, shuffle=True)
        cls.val_dataloader = DataLoader(cls.val_dataset, batch_size=8, shuffle=False)
        cls.model = BiEncoder(model_name="cointegrated/rubert-tiny2", lr=2e-5)
        
    def test_tokenizer_length(self):
        print(len(self.tokenizer))
        assert_that(len(self.tokenizer), equal_to(83828))
        
    def test_train_dataloader(self):
        train_batch = next(iter(self.train_dataloader))
        assert_that(len(train_batch['query_input_ids']), equal_to(8))
        assert_that(train_batch.keys(), equal_to({'query_input_ids', 'query_attention_mask', 'candidate_input_ids', 'candidate_attention_mask'}))
        
    def test_validation_dataloader(self):
        val_batch = next(iter(self.val_dataloader))
        assert_that(len(val_batch['query_input_ids']), equal_to(8))
        assert_that(val_batch.keys(), equal_to({'query_input_ids', 'query_attention_mask', 'candidate_input_ids', 'candidate_attention_mask'}))
        
    def test_embeddings_size(self):
        sample_batch = next(iter(self.train_dataloader))
        query_embeddings, candidate_embeddings = self.model(
            sample_batch['query_input_ids'],
            sample_batch['query_attention_mask'],
            sample_batch['candidate_input_ids'],
            sample_batch['candidate_attention_mask']
        )
        assert query_embeddings.size() == torch.Size([8, 312])
        assert candidate_embeddings.size() == torch.Size([8, 312])
