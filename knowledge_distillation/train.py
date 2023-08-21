import logging
import random
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as opt
from tqdm.auto import trange
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer

from utils.data_preparation import make_pairs, train_val_split
from utils.read_config import HParams, get_hparams
from utils.read_jsonl_data import read_jsonl_data


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def main(hpararms: HParams):
    logger = get_logger(hpararms.experiment_path)
    logger.info(hpararms)

    logger.info("Datasets loading...")
    datasets = [read_jsonl_data(Path(path)) for path in hpararms.data.datasets]

    logger.info("Datasets spliting...")
    data_train: List[Dict[str, List[Dict[str, Any]]]] = list()
    data_val: List[Dict[str, List[Dict[str, Any]]]] = list()
    for dataset in datasets:
        dataset_train, dataset_val = train_val_split(
            dataset, hpararms.data.val_size, True
        )
        data_train.extend(dataset_train)
        data_val.extend(dataset_val)
    logger.info(f"Train dialogs: {len(data_train)} Val dialogs: {len(data_val)}")

    tokenizer, model, optimizer = get_model(hpararms, logger)

    logger.info("Datasets preparation...")
    train_df = make_pairs(
        data_train,
        tokenizer,
        hpararms.train.max_history_tokens,
        hpararms.train.max_history_messages,
    )
    val_df = make_pairs(
        data_val,
        tokenizer,
        hpararms.train.max_history_tokens,
        hpararms.train.max_history_messages,
    )
    train_df = pd.DataFrame([("</s>".join(p[0]), p[1]) for p in train_df])
    val_df = pd.DataFrame([("</s>".join(p[0]), p[1]) for p in val_df])

    train_df = train_df[~train_df.duplicated()]
    train_df = train_df[~(train_df[0].isin(val_df[0]) & train_df[1].isin(val_df[1]))]
    val_df = val_df.drop_duplicates()
    logger.info(f"Train pairs: {len(train_df)} Val pairs: {len(val_df)}")

    best_model = train(
        train_df,
        val_df,
        hpararms.train.batch_size,
        hpararms.train.report_steps,
        hpararms.train.save_steps,
        hpararms.train.epochs,
        tokenizer,
        model,
        optimizer,
        logger,
    )
    save_model(best_model, Path(hpararms.experiment_path) / "checkpoints/best_model/")


def get_logger(model_dir, filename="train.log"):
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)

    logger.addHandler(h)
    return logger


def get_model(
    hparams: HParams, logger: logging.Logger
) -> Tuple[T5Tokenizer, T5ForConditionalGeneration]:
    logger.info("Model loading...")
    t5_config = getattr(hparams.model, "t5_config", None)
    huggingface_path = getattr(hparams.model, "huggingface_path", None)
    assert t5_config or huggingface_path

    tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer.huggingface_path)

    if huggingface_path:
        model = T5ForConditionalGeneration.from_pretrained(
            hparams.model.huggingface_path
        )
    elif t5_config:
        t5_config = T5Config(**hparams.model.t5_config)
        model = T5ForConditionalGeneration(t5_config)

    model = model.cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hparams.train.lr, weight_decay=hparams.train.weight_decay
    )

    return tokenizer, model, optimizer


def train(
    pairs_train: pd.DataFrame,
    pairs_val: pd.DataFrame,
    batch_size: int,
    report_steps: int,
    save_steps: int,
    epochs: int,
    tokenizer: T5Tokenizer,
    model: T5ForConditionalGeneration,
    optimizer: opt.Optimizer,
    logger: logging.Logger,
) -> T5ForConditionalGeneration:
    logger.info("Training starts")
    model.train()
    losses = []

    best_model = model
    best_loss = 1000000

    for epoch in range(epochs):
        logger.info(f"EPOCH {epoch}")
        pairs_train = pairs_train.sample(frac=1)
        for i in trange(0, int(len(pairs_train) / batch_size), leave=False, position=0):
            batch = pairs_train.values[i * batch_size:(i + 1) * batch_size]

            x = tokenizer([p[0] for p in batch], return_tensors="pt", padding="longest")
            x = {k: v.to(model.device, non_blocking=True) for k, v in x.items()}
            # x = x.to(model.device, non_blocking=True)

            y = tokenizer([p[1] for p in batch], return_tensors="pt", padding="longest")
            y = {k: v.to(model.device, non_blocking=True) for k, v in y.items()}
            # y = y.to(model.device, non_blocking=True)

            # -100 - специальное значение, позволяющее не учитывать токены
            # y.input_ids[y.input_ids == 0] = -100
            y["input_ids"][y["input_ids"] == 0] = -100

            loss: torch.Tensor = model(
                input_ids=x["input_ids"],
                attention_mask=x["attention_mask"],
                labels=y["input_ids"],
                decoder_attention_mask=y["attention_mask"],
                return_dict=True,
            ).loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

            if i % report_steps == 0:
                val_loss = np.round(
                    eval(pairs_val, batch_size, tokenizer, model, logger), 3
                )
                train_loss = np.round(np.mean(losses[-report_steps:]), 3)

                step = epoch * int(len(pairs_train) / batch_size) + i
                logger.info(
                    f"step {step} | train loss {train_loss} | val loss {val_loss}"
                )

                if val_loss < best_loss:
                    best_model = model
                    best_loss = val_loss
                model.train()

            if i % save_steps == 0:
                step = epoch * int(len(pairs_train) / batch_size) + i
                save_model(
                    model, Path(hpararms.experiment_path) / f"checkpoints/{step}_steps/"
                )
    return best_model


@torch.no_grad()
def eval(
    pairs: pd.DataFrame,
    batch_size: int,
    tokenizer: T5Tokenizer,
    model: T5ForConditionalGeneration,
    logger: logging.Logger,
) -> float:
    eval_losses = list()
    model.eval()

    pairs = pairs.sample(frac=1)
    for i in trange(0, int(len(pairs) / batch_size), leave=False, position=1):
        batch = pairs.values[i * batch_size:(i + 1) * batch_size]

        x = tokenizer([p[0] for p in batch], return_tensors="pt", padding="longest")
        x = {k: v.to(model.device, non_blocking=True) for k, v in x.items()}
        # x = x.to(model.device)

        y = tokenizer([p[1] for p in batch], return_tensors="pt", padding="longest")
        y = {k: v.to(model.device, non_blocking=True) for k, v in y.items()}
        # y = y.to(model.device)

        # -100 - специальное значение, позволяющее не учитывать padding
        # y.input_ids[y.input_ids == 0] = -100
        y["input_ids"][y["input_ids"] == 0] = -100

        loss: torch.Tensor = model(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"],
            labels=y["input_ids"],
            decoder_attention_mask=y["attention_mask"],
            return_dict=True,
        ).loss
        eval_losses.append(loss.item())

    return np.mean(eval_losses)


def save_model(model: T5ForConditionalGeneration, save_path: Path):
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)


if __name__ == "__main__":
    hpararms = get_hparams()
    main(hpararms)
