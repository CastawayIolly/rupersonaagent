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
        '--save_name',
        type=str,
        default='rubert-base-retriever',
        help='Save path'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='../pretrained_models',
        help='Path where the model is placed'
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='rubert-base',
        help='Model name. Can be the model name from Huggingface Hub'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=6,
        help='Batch size used during training'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of workers in dataloaders'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=3,
        help='Total number of epochs during trinaing'
    )
    parser.add_argument(
        '--accumulation_steps',
        type=int,
        default=8,
        help='Number of accumulation steps during training'
    )
    parser.add_argument(
        "--context_max_length",
        type=int,
        default=32,
        help='Maximum length of context in tokens'
    )
    parser.add_argument(
        "--passage_max_length",
        type=int,
        default=32,
        help='Maximum length of each candidate in tokens'
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help='Learning rate'
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help=''
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help=''
    )

    return parser.parse_args()


def main(args):
    torch.set_float32_matmul_precision('high')
    data_dir = os.path.join(os.curdir, 'data')
    sbquad_path = 'sberquad'
    miracl_path = 'miracl'
    wizint_path = '../internet_model/data/wizint_rus'

    save_dir = os.path.join(os.curdir, 'models')
    save_path = os.path.join(save_dir, args.save_name)

    train_sbquad = dataset.InternetDataset(
        os.path.join(data_dir, sbquad_path, 'train.jsonl'))
    valid_sbquad = dataset.InternetDataset(
        os.path.join(data_dir, sbquad_path, 'valid.jsonl'))

    train_miracl = dataset.InternetDataset(
        os.path.join(data_dir, miracl_path, 'train.jsonl'))
    valid_miracl = dataset.InternetDataset(
        os.path.join(data_dir, miracl_path, 'valid.jsonl'))

    train_wizint = dataset.InternetDataset(os.path.join(wizint_path, 'train_f.jsonl'), wizint_data=True)
    """valid_wizint = dataset.InternetDataset(
        os.path.join(wizint_path, 'valid.jsonl'),
        wizint_data=True)"""

    train = torch.utils.data.ConcatDataset([train_wizint, train_sbquad, train_miracl])
    valid = torch.utils.data.ConcatDataset([valid_sbquad, valid_miracl])

    args.num_training_steps = len(train) * args.num_epochs // (args.accumulation_steps * args.batch_size) + 1

    model = Retriever(
        args.model_dir,
        args.model_name
    )
    model.set_training_params(
        args.lr,
        args.weight_decay,
        args.num_warmup_steps,
        args.num_training_steps,
        args.context_max_length,
        args.passage_max_length
    )

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda data=train, tokenizer=model.tokenizer,
            context_max_length=args.context_max_length,
            passage_max_length=args.passage_max_length:
                dataset.collate_fn(
                    data,
                    tokenizer,
                    context_max_length,
                    passage_max_length
                )
    )

    valid_loader = torch.utils.data.DataLoader(
        valid,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda data=valid, tokenizer=model.tokenizer,
            context_max_length=args.context_max_length,
            passage_max_length=args.passage_max_length: dataset.collate_fn(data, tokenizer, context_max_length, passage_max_length)
    )

    version = f"Train Retriever {args.model_name}. \
        Context length={args.context_max_length}. \
        Passage length={args.passage_max_length}"
    logger = pl_loggers.TensorBoardLogger(save_dir="logs/", version=version)

    trainer = pl.Trainer(
        accelerator='gpu',
        accumulate_grad_batches=args.accumulation_steps,
        devices=1,
        enable_checkpointing=False,
        logger=logger,
        precision="16-mixed",
        max_epochs=args.num_epochs
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    model.M.save_pretrained(save_path)
    model.tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    args = add_cmdline_args()

    main(args)
