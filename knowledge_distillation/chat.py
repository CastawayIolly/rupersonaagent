import argparse
import os
from typing import Tuple

import torch
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        "--max-history-messages",
        default=4,
    )
    return parser


def main(
    tokenizer_pretrained_path: str,
    model_pretrained_path: str,
    device: str,
    max_history_messages: int,
):
    print("Model loading...")
    tokenizer, model = get_model(tokenizer_pretrained_path, model_pretrained_path)
    model = model.eval().to(device)
    print("Bot ready!")

    history_text = list()
    user_text = input("User> ").strip()
    history_text.append(user_text)

    while user_text.lower() not in {"q", "quit", "пока"}:
        history_tmp = "</s>".join(history_text[-max_history_messages:])

        answer = generate_answer(history_tmp, tokenizer, model)
        print(f" Bot> {answer}")
        history_text.append(answer)

        user_text = input("User> ").strip()
        history_text.append(user_text)


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
        args.tokenizer_pretrained_path,
        args.model_pretrained_path,
        args.device,
        args.max_history_messages,
    )
