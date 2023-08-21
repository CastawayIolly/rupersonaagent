import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jsonlines
import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from tqdm.auto import tqdm
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from utils.data_preparation import make_pairs
from utils.read_jsonl_data import read_jsonl_data

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets-path", default="data/Толока Персона Чат")
    parser.add_argument(
        "--tokenizer-pretrained-path", default="cointegrated/rut5-small-chitchat"
    )
    parser.add_argument(
        "--model-pretrained-path",
        default="experiments/exp2-t5-small-chitchat-finetuning/checkpoints/2000_steps",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-history-tokens", default=512)
    parser.add_argument("--max-history-messages", default=4)
    parser.add_argument(
        "--save-path",
        default="data/t5-small-chitchat-finetuned-generation/dialogs.jsonl",
    )
    return parser


def main(
    datasets_path: List[str],
    tokenizer_pretrained_path: str,
    model_pretrained_path: str,
    device: str,
    max_history_tokens: int,
    max_history_messages: int,
    save_path: str,
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

    BLEUs = list()
    new_data = list()

    # Группируем по одинаковым запросам
    for df_tmp in tqdm(df.groupby(by=0)):
        _, df_tmp = df_tmp

        references = [
            tokenizer.tokenize(reference) for reference in df_tmp[1].to_list()
        ]
        for history in df_tmp[0]:
            answer = generate_answer(history, tokenizer, model)
            answer = tokenizer.tokenize(answer)

            bleu = sentence_bleu(
                references, answer, smoothing_function=SmoothingFunction().method1
            )

            BLEUs.append(bleu)
            new_data.append((history, tokenizer.convert_tokens_to_string(answer)))

    print("Generated BLUE-4:", np.mean(BLEUs))
    new_data = pd.DataFrame(new_data)

    dialogs = pairs2dialogs(new_data)
    with jsonlines.open(save_path, mode="w") as writer:
        for dialog in dialogs:
            writer.write(dialog)


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


def pairs2dialogs(pairs: pd.DataFrame) -> List[Dict[str, List[Dict[str, Any]]]]:
    dialogs: List[Dict[str, List[Dict[str, Any]]]] = list()
    for history, answer in tqdm(pairs.values):
        dialog: Dict[str, List[Dict[str, Any]]] = {
            "persons": list(),
            "dialog": list(),
        }
        for text_idx, text in enumerate(history.split("</s>")):
            message = {"person": text_idx % 2, "text": text, "gk": []}
            dialog["dialog"].append(message)
        message = {"person": (text_idx + 1) % 2, "text": answer, "gk": []}
        dialog["dialog"].append(message)
        dialogs.append(dialog)
    return dialogs


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    datasets_path = args.datasets_path
    tokenizer_pretrained_path = args.tokenizer_pretrained_path
    model_pretrained_path = args.model_pretrained_path
    device = args.device
    max_history_tokens = args.max_history_tokens
    max_history_messages = args.max_history_messages
    save_path = args.save_path

    main(
        datasets_path,
        tokenizer_pretrained_path,
        model_pretrained_path,
        device,
        max_history_tokens,
        max_history_messages,
        save_path,
    )
