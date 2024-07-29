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
from torch.utils.data import DataLoader, SequentialSampler
from torchtext.data.metrics import bleu_score

import FiD_dialogue.src.slurm
import FiD_dialogue.src.util
from FiD_dialogue.src.options import Options
import FiD_dialogue.src.data
import FiD_dialogue.src.evaluation
import FiD_dialogue.src.model


def evaluate(model, dataset, dataloader, tokenizer, opt):
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    if opt.write_crossattention_scores:
        model.overwrite_forward_crossattention()
        model.reset_score_storage()
    total = 0
    exactmatch = []
    bleu = []
    if opt.write_results:
        write_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        fw = open(write_path / ('%d.txt' % opt.global_rank), 'a')
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask) = batch

            if opt.write_crossattention_scores:
                model.reset_score_storage()

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=50,
            )

            if opt.write_crossattention_scores:
                crossattention_scores = model.get_crossattention_scores(
                    context_mask.cuda())

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                example = dataset.data[idx[k]]
                if 'answers' in example:
                    score = FiD_dialogue.src.evaluation.ems(ans, example['answers'])
                    exactmatch.append(score)
                    bleu.append(bleu_score([ans.lower().split()], [
                                [example['answers'][0].lower().split()]]))

                if opt.write_results:
                    fw.write(str(example['id']) + "\t" + ans + '\n')
                if opt.write_crossattention_scores:
                    for j in range(context_ids.size(1)):
                        example['ctxs'][j]['score'] = crossattention_scores[k, j].item()

                total += 1
            if (i + 1) % opt.eval_print_freq == 0:
                log = f'Process rank:{opt.global_rank}, {i+1} / {len(dataloader)}'
                if len(exactmatch) == 0:
                    log += '| no answer to compute scores'
                else:
                    log += f' | average = {np.mean(exactmatch):.3f}'
                    log += f' | bleu = {np.mean(bleu):.3f}'
                logger.warning(log)

    logger.warning(
        f'Process rank:{opt.global_rank}, total {total} | average = {np.mean(exactmatch):.3f}')
    if opt.is_distributed:
        torch.distributed.barrier()
    score, total = FiD_dialogue.src.util.weighted_average(np.mean(exactmatch), total, opt)

    bleu_mean = np.mean(bleu)

    return score, bleu_mean, total


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()
    FiD_dialogue.src.slurm.init_distributed_mode(opt)
    FiD_dialogue.src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    dir_path = Path(opt.checkpoint_dir) / opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if opt.write_results:
        (dir_path / 'test_results').mkdir(parents=True, exist_ok=True)
    logger = FiD_dialogue.src.util.init_logger(
        opt.is_main, opt.is_distributed, Path(
            opt.checkpoint_dir) / opt.name / 'run.log')
    if not directory_exists and opt.is_main:
        options.print_options(opt)

    tokenizer = transformers.T5Tokenizer.from_pretrained(
        opt.base_model_path, truncation_side="right")

    collator_function = FiD_dialogue.src.data.Collator(
        opt.text_maxlength,
        tokenizer,
        answer_maxlength=opt.answer_maxlength,
        last_n=opt.last_n)
    eval_examples = FiD_dialogue.src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        # use the global rank and world size attibutes to split the eval set on
        # multiple gpus
        world_size=opt.world_size
    )
    eval_dataset = FiD_dialogue.src.data.Dataset(
        eval_examples,
        opt.n_context,
    )

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=opt.per_gpu_batch_size,
        num_workers=20,
        collate_fn=collator_function
    )

    model_class = FiD_dialogue.src.model.FiDT5
    model = model_class.from_pretrained(opt.model_path)
    model = model.to(opt.device)

    logger.info("Start eval")
    exactmatch, bleu_mean, total = evaluate(
        model, eval_dataset, eval_dataloader, tokenizer, opt)

    logger.info(
        f'EM {100*exactmatch:.2f}, bleu: {bleu_mean}, Total number of example {total}')

    if opt.write_results and opt.is_main:
        glob_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        write_path = Path(opt.checkpoint_dir) / opt.name / 'final_output.txt'
        FiD_dialogue.src.util.write_output(glob_path, write_path)
    if opt.write_crossattention_scores:
        src.util.save_distributed_dataset(eval_dataset.data, opt)
