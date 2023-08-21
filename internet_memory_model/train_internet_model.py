import os
from argparse import ArgumentParser

import in_dataset
import pytorch_lightning as pl
import torch
import torch.utils.data
from in_model import InternetModel
from pytorch_lightning import loggers as pl_loggers


def add_cmdline_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--save_name',
        type=str,
        default='f-t5-big-internet',
        help='Save path where LoRA Adapter will be placed'
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='f-t5-big',
        help='Model name. Can be a name from Huggingface Hub'
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default='pretrained_models',
        help='Path to folder with pretrained model'
    )
    parser.add_argument(
        "--context_max_length",
        type=int,
        default=400,
        help='Maximum length of context in search and response tasks'
    )
    parser.add_argument(
        "--passage_max_length",
        type=int,
        default=60,
        help='Maximum length of passages in knowledge task'
    )
    parser.add_argument(
        "--labels_max_length",
        type=int,
        default=60,
        help='Maximum length of labels in tokens'
    )
    parser.add_argument(
        "--passage_count",
        type=int,
        default=8,
        help="Number of passages per 1 context sample in knowledge task"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=20,
        help='Number of samples per batch used during training'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of workers for DataLoader'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=2,
        help='Number of epochs in training process'
    )
    parser.add_argument(
        '--accumulation_steps',
        type=int,
        default=4,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=25
    )

    return parser.parse_args()


def main(args):
    data_dir = os.path.join(os.curdir, 'data')
    toloka_path = 'toloka'
    ruswiz_path = 'wizint_rus'
    sberquad_path = 'sberquad'

    save_dir = os.path.join(os.curdir, 'models')
    save_path = os.path.join(save_dir, args.save_name)

    search = "search"
    knowledge = "knowledge"
    response = "response"

    train_sbquad = in_dataset.InternetDataset(
        os.path.join(data_dir, sberquad_path, 'train.jsonl'),
        is_toloka=True, skip_tasks=[search, response],
        passage_count=args.passage_count,
        short_replies_len=1, batch_size=args.batch_size)
    valid_sbquad = in_dataset.InternetDataset(
        os.path.join(data_dir, sberquad_path, 'valid.jsonl'),
        is_toloka=True, skip_tasks=[search, response],
        passage_count=args.passage_count,
        short_replies_len=1)

    train_toloka = in_dataset.InternetDataset(
        os.path.join(data_dir, toloka_path, 'train.jsonl'),
        is_toloka=True, skip_tasks=[search, knowledge],
        passage_count=args.passage_count,
        short_replies_len=1, batch_size=args.batch_size)
    valid_toloka = in_dataset.InternetDataset(
        os.path.join(data_dir, toloka_path, 'valid.jsonl'),
        is_toloka=True, skip_tasks=[search, knowledge],
        passage_count=args.passage_count,
        short_replies_len=1)

    train_ruswiz = in_dataset.InternetDataset(
        os.path.join(data_dir, ruswiz_path, 'train_f.jsonl'),
        is_toloka=False, skip_tasks=[],
        passage_count=args.passage_count,
        short_replies_len=1, batch_size=args.batch_size)
    """valid_ruswiz = in_dataset.InternetDataset(
        os.path.join(data_dir, ruswiz_path, 'valid.jsonl'),
        is_toloka=False, skip_tasks=[],
        passage_count=args.passage_count,
        short_replies_len=1)"""

    train = torch.utils.data.ConcatDataset([train_ruswiz, train_toloka, train_sbquad])
    valid = torch.utils.data.ConcatDataset([valid_sbquad, valid_toloka])

    args.num_training_steps = len(train_ruswiz) * args.num_epochs // (args.accumulation_steps * args.batch_size) + 1

    model = InternetModel(
        args.model_dir,
        args.model_name,
        args.save_name,
        args.labels_max_length
    )
    model.add_lora_adapter()
    model.M.make_fid_encoder()
    model.set_training_args(args)

    args.tokenizer = model.tokenizer

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=False,
        batch_sampler=in_dataset.BatchSampler(train, args.batch_size),
        num_workers=args.num_workers,
        collate_fn=lambda data=train, args=args: in_dataset.collate_fn(data, args))

    valid_loader = torch.utils.data.DataLoader(
        valid,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda data=valid, args=args: in_dataset.collate_fn(data, args))

    version = f"Train {args.model_name}. \
        Context length={args.context_max_length}. \
        Passage length={args.passage_max_length}. \
        Passage count={args.passage_count}"
    logger = pl_loggers.TensorBoardLogger(save_dir="logs/", version=version)

    trainer = pl.Trainer(
        accelerator='gpu',
        accumulate_grad_batches=args.accumulation_steps,
        devices=1,
        enable_checkpointing=False,
        logger=logger,
        precision="bf16-mixed",
        max_epochs=args.num_epochs,
        num_sanity_val_steps=10,
        log_every_n_steps=5,
        detect_anomaly=True
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader)

    model.M.make_base_encoder()

    model.M.save_pretrained(save_path)
    model.tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    args = add_cmdline_args()

    main(args)
