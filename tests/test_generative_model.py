import torch
import pytest
import transformers
import json

from generative_model.model import T5MultiTask
from generative_model.data_module import TolokaDataModule

from hamcrest import assert_that, equal_to


@pytest.mark.generative
class TestGenerativeModel:
    self.tokenizer = transformers.AutoTokenizer.from_pretrained("cointegrated/rut5-base-multitask",
                                                                truncation_side='left',
                                                                padding_side='right')
    self.t5 = transformers.T5ForConditionalGeneration.from_pretrained("cointegrated/rut5-base-multitask",
                                                                      resume_download=True)

    with open('/home/stc/persona/data/preprocessing/spec_tokens.json') as spec_tokens_config:
        spec_tokens = json.load(spec_tokens_config)
    self.tokenizer.add_special_tokens(
        {"additional_special_tokens": [spec_tokens[k] for k in spec_tokens]}
    )
    self.datamodule = TolokaDataModule(
        data_dir='/home/stc/persona/data',
        datasets=['current_gk', 'next_answer'],
        tokenizer=self.tokenizer,
        spec_tokens=spec_tokens,
        train_batch_size=128,
        val_batch_size=256,
        test_batch_size=256,
    )
    self.model = T5MultiTask(
        model=self.t5,
        datamodule=self.datamodule,
        lr=5e-5,
        num_warmup_steps=1000,
        pooling="mean",
        distance="cosine",
        scale=20,
        train_batch_size=128,
        val_batch_size=256,
        test_batch_size=256,
    )

    def test_dictionary(self):
        assert_that(len(self.tokenizer), equal_to(30015))

    def testdatamodule(self):
        train = self.datamodule.train_dataloader()
        assert_that(len(train), equal_to(2231))
        assert_that(train.column_names, equal_to(['current_gk', 'task', 'next_answer']))
        assert_that(list(train[0].keys()), equal_to(['current_gk', 'task', 'next_answer']))

    def test_embeddings_size(self):
        inp = self.tokenizer(['тест'], return_tensors='pt')
        self.model.get_embedding(inp).size() == torch.Size([1, 768])
