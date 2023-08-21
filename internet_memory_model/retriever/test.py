import os
from argparse import ArgumentParser

import dataset
import pytorch_lightning as pl
import torch
import torch.utils.data
from model import Retriever
from pytorch_lightning import loggers as pl_loggers


def add_cmdline_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--model_dir',
        type=str,
        default='models',
        help='Path where the model is placed'
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='rubert-base-retriever',
        help='Model name. Can be the model name from Huggingface Hub'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size used during training'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of workers in dataloaders'
    )
    parser.add_argument(
        "--context_max_length",
        type=int,
        default=64,
        help='Maximum length of context in tokens'
    )
    parser.add_argument(
        "--passage_max_length",
        type=int,
        default=128,
        help='Maximum length of each candidate in tokens'
    )

    return parser.parse_args()


def main(args):
    data_dir = os.path.join(os.curdir, 'data')
    sbquad_path = 'sberquad'
    miracl_path = 'miracl'

    valid_sbquad = dataset.InternetDataset(os.path.join(data_dir, sbquad_path, 'valid.jsonl'))
    valid_miracl = dataset.InternetDataset(os.path.join(data_dir, miracl_path, 'valid.jsonl'))
    """valid_wizint = dataset.InternetDataset(
        os.path.join(wizint_path, 'valid.jsonl'),
        wizint_data=True)"""

    valid = torch.utils.data.ConcatDataset([valid_sbquad, valid_miracl])

    model = Retriever(
        args.model_dir,
        args.model_name
    )

    valid_loader = torch.utils.data.DataLoader(
        valid,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda data=valid, tokenizer=model.tokenizer, context_max_length=args.context_max_length,
                          passage_max_length=args.passage_max_length: dataset.collate_fn(data, tokenizer, context_max_length, passage_max_length)
    )

    version = f"Validation Retriever {args.model_name}. \
        Context length={args.context_max_length}. \
        Passage length={args.passage_max_length}"
    logger = pl_loggers.TensorBoardLogger(save_dir="logs/", version=version)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        enable_checkpointing=False,
        logger=logger,
    )

    trainer.validate(model, dataloaders=valid_loader)


if __name__ == "__main__":
    args = add_cmdline_args()

    main(args)
