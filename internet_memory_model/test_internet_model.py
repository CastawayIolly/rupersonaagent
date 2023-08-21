import os
from argparse import ArgumentParser

import in_dataset
import pytorch_lightning as pl
import torch
import torch.utils.data
import in_model as im
from pytorch_lightning import loggers as pl_loggers


def add_cmdline_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--test_dir',
        type=str,
        default='own_data',
        help='Path to folder with test dataset'
    )
    parser.add_argument(
        '--test_file',
        type=str,
        default='test_dataset.jsonl',
        help='Name of test dataset file'
    )
    parser.add_argument(
        '--save_name',
        type=str,
        default='f-t5-big-internet',
        help='Path to trained LoRA Adapter'
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
        '--num_workers',
        type=int,
        default=6,
        help='Number of workers for DataLoader'
    )
    parser.add_argument(
        "--context_max_length",
        type=int,
        default=2048,
        help='Maximum length of context in search and response tasks'
    )
    parser.add_argument(
        "--passage_max_length",
        type=int,
        default=64,
        help='Maximum length of passages in knowledge task'
    )
    parser.add_argument(
        "--labels_max_length",
        type=int,
        default=128,
        help='Maximum length of labels in tokens'
    )
    parser.add_argument(
        "--passage_count",
        type=int,
        default=10,
        help="Number of passages per 1 context sample in knowledge task"
    )

    return parser.parse_args()


def main(args):
    data_dir = os.path.join(os.curdir, 'data')

    test_skip_tasks = []

    if "toloka" or "own_data" in args.test_dir:
        test_skip_tasks = ["search"]
    if "wizint" in args.test_dir:
        test_skip_tasks = []

    test = in_dataset.InternetDataset(
        os.path.join(data_dir, args.test_dir, args.test_file),
        is_toloka="wizint" not in args.test_dir,
        skip_tasks=test_skip_tasks,
        passage_count=args.passage_count
    )

    model = im.InternetModel(
        args.model_dir,
        args.model_name,
        args.save_name,
        args.labels_max_length
    )
    model.load_lora_adapter()
    model.M.make_fid_encoder()

    args.tokenizer = model.tokenizer

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda data=test, args=args: in_dataset.collate_fn(data, args))

    version = f"{args.model_name}.{args.test_dir}.{args.labels_max_length}.{args.passage_max_length}.{args.passage_count}"
    logger = pl_loggers.TensorBoardLogger(save_dir="logs/", version=version)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        enable_checkpointing=False,
        logger=logger
    )

    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    args = add_cmdline_args()

    main(args)
