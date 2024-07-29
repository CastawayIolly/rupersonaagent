import os
from typing import Literal

import lightning.pytorch as pl
import torch
import transformers
import torchmetrics

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MyModel(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        lr: float = 5e-5,
        num_warmup_steps: int = 1000,
        pooling: Literal["mean", "cls"] = "mean",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[join]"]})

        # encoder
        self.encoder = transformers.AutoModel.from_pretrained(
            self.hparams.model_name_or_path
        )
        self.encoder.resize_token_embeddings(
            len(self.tokenizer), pad_to_multiple_of=8
        )

        # dropout & activations
        self.dropout = torch.nn.Dropout(p=0.2, inplace=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)

        # classification heads
        self.head = torch.nn.Linear(
            self.encoder.config.hidden_size, 7
        )
        # loss
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 100.0, 150.0, 150.0, 20.0, 70.0, 70.0]))

    def configure_model(self):
        binary_classification = torchmetrics.MetricCollection(
            {
                "f1": torchmetrics.F1Score(task="multiclass", num_classes=7, average="none"),
            }
        )
        self.metrics = {}
        for split in ["train", "val", "test"]:
            self.metrics[split] = binary_classification.clone(
                prefix=f"{split}/"
            )

    def forward(self, input_ids, attention_mask, mask):
        # encode
        model_output = self.encoder(input_ids, attention_mask=attention_mask)
        # pool
        model_output = model_output[0][mask]

        logits = self.head(model_output)
        return logits

    def training_step(self, batch: dict, batch_idx):
        # Compute loss
        logits = self(batch["input_ids"], batch["attention_mask"], batch['mask'])
        labels = batch["label"]
        loss = self.cross_entropy_loss(logits, labels.to(torch.long))

        # Calculate metrics
        metrics = self.metrics["train"](
            logits, labels
        )
        metrics = {f"train/f1{i}": m for i, m in enumerate(metrics['train/f1'])}
        # Log
        self.log("train_loss", loss.item(), sync_dist=True, batch_size=labels.shape[0])
        self.log_dict(
            metrics,
            sync_dist=True,
            batch_size=labels.shape[0],
            on_step=False,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch: dict, batch_idx):
        # Compute loss
        logits = self(batch["input_ids"], batch["attention_mask"], batch['mask'])
        labels = batch["label"]
        loss = self.cross_entropy_loss(logits, labels.to(torch.long))

        # Calculate metrics
        metrics = self.metrics["val"](
            logits, labels
        )
        metrics = {f"val/f1{i}": m for i, m in enumerate(metrics['val/f1'])}

        # Log
        self.log("val_loss", loss.item(), sync_dist=True, batch_size=labels.shape[0])
        self.log_dict(
            metrics,
            sync_dist=True,
            batch_size=labels.shape[0],
            on_step=False,
            on_epoch=True,
        )

        return loss

    def test_step(self, batch: dict, batch_idx, dataloader_idx=0):
        # Compute loss
        logits = self(batch["input_ids"], batch["attention_mask"], batch['mask'])
        labels = batch["label"]
        loss = self.cross_entropy_loss(logits, labels.to(torch.long))

        # Calculate metrics
        metrics = self.metrics["test"](
            logits, labels
        )
        metrics = {f"test/f1{i}": m for i, m in enumerate(metrics['test/f1'])}

        # Log
        self.log("test_loss", loss.item(), sync_dist=True, batch_size=labels.shape[0])
        self.log_dict(
            metrics,
            sync_dist=True,
            batch_size=labels.shape[0],
            on_step=False,
            on_epoch=True,
        )
        for text, target, pred in zip(batch["input_ids"], batch["label"], logits.argmax(dim=1)):
            print(target.item(), pred.item(), self.tokenizer.decode(text))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [
            {"scheduler": scheduler, "name": "cosine_scheduler", "interval": "step"}
        ]

    @staticmethod
    def cls_pooling(model_output: torch.Tensor):
        return model_output[0][:, 0]

    # https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    @staticmethod
    def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.shape).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
