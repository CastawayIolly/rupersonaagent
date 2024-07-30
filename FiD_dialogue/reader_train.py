# This source code has been adapted from original FiD implementation by
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# The original repository: https://github.com/facebookresearch/FiD
# The original code is licensed by Attribution-NonCommercial 4.0 International (https://creativecommons.org/licenses/by-nc/4.0/)
# This code has been modified to allow for dialogue agents training
# The source code found in this part of the repository is licensed accordingly
# The text of the license can be found in the LICENSE file at the root of this directory

import torch
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from FiD_dialogue.src.options import Options
from torchtext.data.metrics import bleu_score
import FiD_dialogue.src.slurm
import FiD_dialogue.src.util
import FiD_dialogue.src.evaluation
import FiD_dialogue.src.data
import FiD_dialogue.src.model


def train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(
                Path(opt.checkpoint_dir) / opt.name)
        except BaseException:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    # different seed for different sampling depending on global_rank
    torch.manual_seed(opt.global_rank + opt.seed)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=10,
        collate_fn=collator
    )

    curr_loss = 0.0
    epoch = 1
    model.train()
    while step < opt.total_steps:
        epoch += 1

        for i, batch in enumerate(train_dataloader):
            if step % 100 == 0:
                logger.info(f"Step: {step}")

            step += 1
            (idx, labels, _, context_ids, context_mask) = batch

            train_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda()
            )[0]

            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = FiD_dialogue.src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if step % opt.eval_freq == 0:
                logger.info("Eval start")
                dev_em, dev_bleu = evaluate(
                    model, eval_dataset, tokenizer, collator, opt)
                model.train()
                if opt.is_main:
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        FiD_dialogue.src.util.save(
                            model,
                            optimizer,
                            scheduler,
                            step,
                            best_dev_em,
                            opt,
                            checkpoint_path,
                            'best_dev')
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += f"evaluation: {100*dev_em:.2f}EM |"
                    log += f"evaluation: {dev_bleu:.2f} Bleu |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", dev_em, step)
                        tb_logger.add_scalar(
                            "Training", curr_loss / (opt.eval_freq), step)
                    curr_loss = 0.

            if opt.is_main and step % opt.save_freq == 0:
                FiD_dialogue.src.util.save(model, optimizer, scheduler, step, best_dev_em,
                                           opt, checkpoint_path, f"step-{step}")
            if step > opt.total_steps:
                break


def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            sampler=sampler,
                            batch_size=opt.per_gpu_batch_size,
                            drop_last=False,
                            num_workers=10,
                            collate_fn=collator
                            )
    model.eval()
    total = 0
    exactmatch = []
    bleu = []

    model = model.module if hasattr(model, "module") else model

    num_batches = len(dataloader)
    batch_size = opt.per_gpu_batch_size

    example_ids = np.random.choice(
        np.arange(num_batches * batch_size),
        10,
        replace=False)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, target_ids, target_mask, context_ids, context_mask) = batch

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=50
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)

                gold = dataset.get_example(idx[k])['answers']

                if i * batch_size + k in example_ids and opt.output_examples:
                    print("ans:", ans)
                    print("gold:", gold[0])
                    print()

                score = FiD_dialogue.src.evaluation.ems(ans, gold)
                bleu.append(bleu_score([ans.lower().split()], [
                            [gold[0].lower().split()]]))

                total += 1
                exactmatch.append(score)

    exactmatch, total = FiD_dialogue.src.util.weighted_average(
        np.mean(exactmatch), total, opt)
    bleu_mean = np.mean(bleu)

    return exactmatch, bleu_mean


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    FiD_dialogue.src.slurm.init_distributed_mode(opt)
    FiD_dialogue.src.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir) / opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    logger = FiD_dialogue.src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_class = FiD_dialogue.src.model.FiDT5

    # load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(
        opt.base_model_path, truncation_side="right")
    collator = FiD_dialogue.src.data.Collator(
        opt.text_maxlength,
        tokenizer,
        answer_maxlength=opt.answer_maxlength,
        last_n=opt.last_n)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = FiD_dialogue.src.data.load_data(
        opt.train_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    train_dataset = FiD_dialogue.src.data.Dataset(train_examples, opt.n_context)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = FiD_dialogue.src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    eval_dataset = FiD_dialogue.src.data.Dataset(eval_examples, opt.n_context)

    if not checkpoint_exists and opt.model_path == "none":
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(
            opt.base_model_path)
        model = FiD_dialogue.src.model.FiDT5(t5.config)
        model.load_t5(t5.state_dict())
        model = model.to(opt.local_rank)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em = 0, 0.0
    elif opt.model_path == "none":
        load_path = checkpoint_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            FiD_dialogue.src.util.load(model_class, load_path, opt, reset_params=False)
        logger.info(f"Model loaded from {load_path}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            FiD_dialogue.src.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path
    )
