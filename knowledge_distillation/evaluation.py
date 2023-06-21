import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from tqdm.auto import tqdm, trange
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer

from utils.data_preparation import make_pairs
from utils.read_jsonl_data import read_jsonl_data

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets-path",
        default="data/test",
    )
    parser.add_argument(
        "--tokenizer-pretrained-path",
        default="cointegrated/rut5-small-chitchat",
    )
    parser.add_argument(
        "--model-pretrained-path",
        default="experiments/exp2-t5-small-chitchat-finetuning/checkpoints/2000_steps",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parser.add_argument(
        "--max-history-tokens",
        default=1024,
    )
    parser.add_argument(
        "--max-history-messages",
        default=4,
    )
    parser.add_argument(
        "--repeats-count",
        default=3,
    )
    return parser


def main(
    datasets_path: List[str],
    tokenizer_pretrained_path: str,
    model_pretrained_path: str,
    device: str,
    max_history_tokens: int,
    max_history_messages: int,
    repeats_count: int,
):
    datasets = [read_jsonl_data(path) for path in Path(datasets_path).iterdir()]
    data: List[Dict[str, List[Dict[str, Any]]]] = list()
    for dataset in datasets:
        data.extend(dataset)

    tokenizer, model = get_model(tokenizer_pretrained_path, model_pretrained_path)
    model = model.eval().to(device)

    df = make_pairs(data, tokenizer, max_history_tokens, max_history_messages)
    df = pd.DataFrame([("</s>".join(p[0]), p[1]) for p in df])
    df = df[~df.duplicated()]

    generation_times = list()
    total_BLEUs = list()
    for generation_idx in trange(repeats_count, position=0, leave=False):
        BLEUs = list()
        # Группируем по одинаковой истории сообщений
        for df_tmp in tqdm(df.groupby(by=0), position=1, leave=False):
            _, df_tmp = df_tmp

            references = [
                tokenizer.tokenize(reference) for reference in df_tmp[1].to_list()
            ]
            for history in df_tmp[0]:
                # Генерируем ответ и замеряем скорость
                start_time = time.time()
                answer = generate_answer(history, tokenizer, model)
                generation_times.append(time.time() - start_time)

                answer = tokenizer.tokenize(answer)
                bleu = sentence_bleu(
                    references, answer, smoothing_function=SmoothingFunction().method1
                )
                BLEUs.append(bleu)

        mean, std = np.round(np.mean(BLEUs), 4), np.round(np.std(BLEUs), 4)
        print()
        print(f"BLUE-4 №{generation_idx}: {mean}±{std}")
        total_BLEUs.extend(BLEUs)

    mean, std = np.round(np.mean(total_BLEUs), 4), np.round(np.std(total_BLEUs), 4)
    print(f"BLUE-4 total: {mean}±{std}")

    mean = np.round(np.mean(generation_times), 4)
    std = np.round(np.std(generation_times), 4)
    print(f"Average generation time: {mean}±{std}")


def get_model(
    tokenizer_pretrained_path: str, model_pretrained_path: str
) -> Tuple[T5Tokenizer, T5ForConditionalGeneration]:
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_pretrained_path)
    model = T5ForConditionalGeneration.from_pretrained(model_pretrained_path)
    return tokenizer, model


@torch.no_grad()
def generate_answer(
    history_text: str, tokenizer: T5Tokenizer, model: T5ForConditionalGeneration
) -> str:
    inputs = tokenizer(history_text, return_tensors="pt")
    hypotheses = model.generate(
        **{k: v.to(model.device) for k, v in inputs.items()},
        do_sample=True,
        top_p=0.5,
        num_return_sequences=1,
        repetition_penalty=4.5,
        max_length=1024,
    )
    return tokenizer.decode(hypotheses[0], skip_special_tokens=True)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(
        args.datasets_path,
        args.tokenizer_pretrained_path,
        args.model_pretrained_path,
        args.device,
        args.max_history_tokens,
        args.max_history_messages,
        args.repeats_count,
    )
