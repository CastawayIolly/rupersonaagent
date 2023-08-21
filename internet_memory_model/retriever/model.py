import os

import pytorch_lightning as pl
import torch
import torchmetrics
from transformers import (BertModel, AutoTokenizer,
                          get_cosine_schedule_with_warmup)


PERSONA1 = '<Персона пользователя 1>'
PERSONA2 = '<Персона пользователя 2>'
SEARCH_TASK = '<Выполнить поиск>'
KNOWLEDGE_TASK = '<Извлечь знания>'
RESPONSE_TASK = '<Сгенерировать реплику>'
NO_QUERY = '<Нет запроса>'
NO_KNOWLEDGE = '<Нет знаний>'
KNOWLEDGE = '<Знания>'
USER1_REPLY = '<Реплика пользователя 1>'
USER2_REPLY = '<Реплика пользователя 2>'
KNOWLEDGE_RESULT = '<Итоговые знания>'
SEARCH_RESULT = '<Итоговый поиск>'
RESPONSE_RESULT = '<Итоговая реплика>'
QUESTION = '<Вопрос>'
CANDIDATE = '<Кандидат>'

ATTR_TO_SPECIAL_TOKEN = {
    'additional_special_tokens': [
        PERSONA1, PERSONA2,
        USER1_REPLY, USER2_REPLY,
        SEARCH_TASK, KNOWLEDGE_TASK, RESPONSE_TASK,
        KNOWLEDGE, NO_KNOWLEDGE, NO_QUERY,
        KNOWLEDGE_RESULT, SEARCH_RESULT, RESPONSE_RESULT
    ]
}


class Retriever(pl.LightningModule):
    def __init__(
        self,
        model_dir,
        model_name
    ):
        super().__init__()

        model_path = os.path.join(model_dir, model_name)

        if os.path.exists(model_path):  # Load from local path
            self.M = BertModel.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:  # Load from Huggingface Hub
            self.M = BertModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
        self.M.resize_token_embeddings(len(self.tokenizer))

        self.metrics = torchmetrics.MetricCollection(
            {
                'Top@1': TopKAccuracy(topk=1),
                'Top@5': TopKAccuracy(topk=5),
                'Top@20': TopKAccuracy(topk=20),
                'Top@30': TopKAccuracy(topk=30),
            }
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.save_hyperparameters(ignore=["M"])

    def set_training_params(
        self,
        lr,
        weight_decay,
        num_warmup_steps,
        num_training_steps,
        context_max_length,
        passage_max_length
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.context_max_length = context_max_length
        self.passage_max_length = passage_max_length

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def dot(self, x, y):
        return torch.mm(x, y.transpose(0, 1))

    def training_step(self, batch, batch_idx):
        q_ids, q_mask = batch[0], batch[1]
        p_ids, p_mask = batch[2], batch[3]
        labels = batch[4]

        q_pred = self.M(
            input_ids=q_ids,
            attention_mask=q_mask
        ).last_hidden_state[:, 0]

        p_pred = self.M(
            input_ids=p_ids,
            attention_mask=p_mask
        ).last_hidden_state[:, 0]

        similarity = self.dot(q_pred, p_pred)
        softmax_score = torch.nn.functional.log_softmax(similarity, dim=-1)

        loss = self.loss_fn(softmax_score, labels)

        accuracy = self.metrics(softmax_score, labels.long())
        accuracy = {f"Train {k}": accuracy[k] for k in accuracy}

        self.log(
            name='lr',
            value=self.trainer.optimizers[0].param_groups[0]["lr"],
            on_epoch=False, on_step=True
        )
        self.log(
            name='Train loss',
            value=loss,
            on_epoch=True, on_step=True
        )
        self.log_dict(
            accuracy,
            on_epoch=True, on_step=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        q_ids, q_mask = batch[0], batch[1]
        p_ids, p_mask = batch[2], batch[3]
        labels = batch[4]

        q_pred = self.M(
            input_ids=q_ids,
            attention_mask=q_mask
        ).last_hidden_state[:, 0]

        p_pred = self.M(
            input_ids=p_ids,
            attention_mask=p_mask
        ).last_hidden_state[:, 0]

        similarity = self.dot(q_pred, p_pred)
        softmax_score = torch.nn.functional.log_softmax(similarity, dim=-1)

        loss = self.loss_fn(softmax_score, labels)

        accuracy = self.metrics(softmax_score, labels.long())
        accuracy = {f"Validation {k}": accuracy[k] for k in accuracy}

        self.log(
            name='Validation loss',
            value=loss,
            on_epoch=True, on_step=True
        )
        self.log_dict(
            accuracy,
            on_epoch=True, on_step=True
        )

        return loss


class TopKAccuracy(torchmetrics.Metric):
    def __init__(self, topk=1):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.topk = topk

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        if self.topk < preds.shape[1]:
            _, pred = preds.topk(self.topk, dim=1, largest=True, sorted=True)

            self.correct += (target * torch.zeros_like(target).scatter(1, pred[:, :self.topk], 1)).sum()
        else:
            self.correct += target.sum()

        self.total += target.sum()

    def compute(self):
        return self.correct.float() / self.total
