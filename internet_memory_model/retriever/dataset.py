import json
from functools import reduce

import torch
import model
from tqdm.auto import tqdm


USER1_ACTION = "Apprentice => Wizard"
USER2_ACTION = "Wizard => Apprentice"
USER2_SEARCH = "Wizard => SearchAgent"


class InternetDataset(torch.utils.data.Dataset):
    def __init__(self, path, do_preprocess=False, wizint_data=False):
        super().__init__()
        if not wizint_data:
            self.data = get_data(path, do_preprocess)
        else:
            self.data = get_wizint_data(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(
    data,
    tokenizer,
    context_max_length,
    passage_max_length
):
    questions = [i['questions'] for i in data]
    passages = [i['passages'] for i in data]
    labels = [i['labels'] for i in data]

    q = tokenizer(
        [model.QUESTION + model.USER2_REPLY + i for i in questions],
        max_length=context_max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    q_ids, q_mask = q.input_ids, q.attention_mask

    p = tokenizer(
        [model.CANDIDATE + i for i in reduce(lambda x, y: x + y, passages)],
        max_length=passage_max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    p_ids, p_mask = p.input_ids, p.attention_mask

    # Convert labels to shape Q * P
    # P - passages count
    # Q - questions count
    flat_labels = []
    labels_len = [len(i) for i in labels]

    for i, elem in enumerate(labels):
        labels_elem = [0 for _ in range(sum(labels_len[:i]))]
        labels_elem += [int(j) for j in elem]
        labels_elem += [0 for _ in range(sum(labels_len[i + 1:]))]

        flat_labels.append(labels_elem)

    flat_labels = torch.FloatTensor(flat_labels)

    return (q_ids, q_mask, p_ids, p_mask, flat_labels)


def get_data(path, do_preprocess=False):
    result = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)

            if do_preprocess:
                data = preprocess(data)

            q, p, labels = parse(data)

            result.append({
                "questions": q,
                "passages": p,
                "labels": labels
            })

    return result


def get_wizint_data(path):
    result = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print("Json decode error. Skipping...")
                continue
            data = data[list(data.keys())[0]]

            dialog = data['dialog_history']

            context = model.PERSONA1 + data['user1']
            for elem in dialog:
                if elem['context'] != {} and elem['context']['contents'] != [] and 'selected_contents' in elem['context']:
                    labels = reduce(lambda x, y: x + y, elem['context']['selected_contents'][1:])
                    texts = reduce(lambda x, y: x + y, [i['content'] for i in elem['context']['contents']])

                    result.append({
                        "questions": context,
                        "labels": labels,
                        "passages": texts
                    })
                if elem['action'] == USER1_ACTION:
                    context += model.USER1_REPLY + elem['text']
                if elem['action'] == USER2_ACTION:
                    context += model.USER2_REPLY + elem['text']
                if elem['action'] == USER2_SEARCH:
                    context += model.SEARCH_RESULT + elem['text']
    return result


def parse(data):
    question = data['query']
    passages = [i['text'] for i in data['positive_passages']]
    labels = [True for _ in range(len(passages))]

    passages.extend([i['text'] for i in data['negative_passages']])
    labels.extend([False for _ in range(len(passages) - len(labels))])

    return question, passages, labels


def preprocess(data):
    data['query'] = data['query'].replace("\xa0", " ")

    for i in data['positive_passages']:
        i['text'] = i['text'].replace("\xa0", " ")
        i['title'] = i['title'].replace("\xa0", " ")

    for i in data['negative_passages']:
        i['text'] = i['text'].replace("\xa0", " ")
        i['title'] = i['title'].replace("\xa0", " ")

    return data
