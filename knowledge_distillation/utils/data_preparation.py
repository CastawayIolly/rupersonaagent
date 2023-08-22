import random

from typing import Any, Dict, List, Tuple

from tqdm.auto import tqdm
from transformers.models.t5.tokenization_t5 import T5Tokenizer


def train_val_split(
    data: List[Dict[str, str]],
    val_size: int = 0.05,
    shuffle: bool = True,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if shuffle:
        random.shuffle(data)
    val_len = int(val_size * len(data))
    data_train, data_val = data[:-val_len], data[-val_len:]
    return data_train, data_val


def compress_consecutive_statements(dialog: List[Dict[str, Any]]):
    # Сжимаем все подряд идущие высказывания одного спикера
    compressed_dialog: List[Dict[str, Any]] = list()

    last_person: int = dialog[0]["person"]
    whole_text = [dialog[0]["text"]]
    for message in dialog[1:]:
        text, person = message["text"], message["person"]

        if last_person == person:
            whole_text.append(text)
        else:
            new_message = {"person": last_person, "text": " ".join(whole_text)}
            compressed_dialog.append(new_message)
            last_person = person
            whole_text = [text]

    new_message = {"person": last_person, "text": " ".join(whole_text)}
    compressed_dialog.append(new_message)

    return compressed_dialog


def make_pairs(
    data: List[Dict[str, List[Dict[str, Any]]]],
    tokenizer: T5Tokenizer,
    max_history_tokens: int,
    max_history_messages: int = 3,
) -> List[Tuple[str, str]]:
    # Все пары "история общения -> ответ"
    pairs: List[Tuple[str, str]] = list()

    for data_item in tqdm(data):
        # Пары "история общения -> ответ" в рамках одного диалога
        dialog_pairs: List[Tuple[List[str], str]] = list()

        # Сжимаем все подряд идущие высказывания одного спикера
        dialog = compress_consecutive_statements(data_item["dialog"])

        historical_text = [dialog[0]["text"]]
        for message in dialog[1:]:
            text = message["text"]
            for history_messages_len in range(1, max_history_messages + 1):
                if len(historical_text) >= history_messages_len:
                    dialog_pairs.append((historical_text[-history_messages_len:], text))

            offset = 0
            historical_text = dialog_pairs[-1][0][offset:] + [text]

            while (
                len(tokenizer("</s>".join(historical_text)).input_ids) > max_history_tokens
            ):
                offset += 1
                historical_text = dialog_pairs[-1][0][offset:] + [text]

        pairs.extend(dialog_pairs)

    return pairs
