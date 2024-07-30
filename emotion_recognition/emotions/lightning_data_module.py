import lightning.pytorch as pl
import transformers
import torch
import datasets
import datasets.distributed


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str = 'cointegrated/rubert-tiny2',
        data_dir: str = "daily_dialog",
        train_bs: int = 32,
        val_bs: int = 32,
        test_bs: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_bs = train_bs
        self.val_bs = val_bs
        self.test_bs = test_bs

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[join]"]})
        self.collator = Collator(tokenizer=self.tokenizer)

        ds = datasets.load_dataset(data_dir)
        self.ds = ds

    def train_dataloader(self):
        print("\nreturning new train_dataloader\n")

        return torch.utils.data.DataLoader(
            self.ds['train'],
            batch_size=self.train_bs,
            num_workers=12,
            prefetch_factor=1,
            drop_last=True,
            collate_fn=self.collator
        )

    def val_dataloader(self):
        print("\nreturning new val_dataloader\n")

        return torch.utils.data.DataLoader(
            self.ds['validation'],
            batch_size=self.train_bs,
            num_workers=12,
            prefetch_factor=1,
            drop_last=False,
            collate_fn=self.collator
        )

    def test_dataloader(self):
        print("\nreturning new test_dataloader\n")

        return torch.utils.data.DataLoader(
            self.ds['test'],
            batch_size=self.train_bs,
            num_workers=12,
            prefetch_factor=1,
            drop_last=False,
            collate_fn=self.collator
        )


class Collator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.join_token = "[join]"
        self.join_token_id = tokenizer(self.join_token, add_special_tokens=False)['input_ids'][0]

    def __call__(self, batch):
        dialogs = []
        for dialog in batch:
            dialog = self.join_token + self.join_token.join(dialog['dialog'])
            dialogs.append(dialog)
        out_batch = self.tokenizer(
            dialogs,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        mask = out_batch['input_ids'] == self.join_token_id
        out_batch["mask"] = mask
        labels = []
        for m, emotion in zip(mask, batch):
            emotion = emotion['emotion'][:sum(m)]
            labels += emotion
        out_batch['label'] = torch.tensor(labels)
        return out_batch
