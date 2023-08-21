import json
import math
import random
from functools import reduce

import in_model as im
import torch
from torch.utils.data import Sampler
from tqdm.auto import tqdm


class InternetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        is_toloka,
        skip_tasks,
        passage_count,
        short_replies_len=1,
        batch_size=1
    ):
        super().__init__()
        self.data = get_data(
            path,
            is_toloka,
            skip_tasks,
            passage_count,
            short_replies_len,
            batch_size
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_data(
    path,
    is_toloka=True,
    skip_tasks=[],
    passage_count=1,
    short_replies_len=1,
    batch_size=1
):
    if is_toloka:
        persona1 = 'persona1'
        persona2 = 'persona2'
        user1_action = 'user1'
        user2_action = 'user2'
        search_action = None
        engine_action = None
    else:
        persona1 = 'apprentice_persona'
        persona2 = None
        user1_action = 'Apprentice => Wizard'
        user2_action = 'Wizard => Apprentice'
        search_action = 'Wizard => SearchAgent'
        engine_action = 'SearchAgent => Wizard'

    keys = {
        'persona1': persona1,
        'persona2': persona2,
        'user1': user1_action,
        'user2': user2_action,
        'search': search_action,
        'engine': engine_action
    }

    result = []

    print(f'Loading from path: {path}')

    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                data = json.loads(line)
            except json.decoder.JSONDecodeError:
                print("Json error happend. Skipping...")
                continue
            data = list(data.values())[0]

            result.extend(parse_data(data, keys, passage_count, short_replies_len))

    random.shuffle(result)
    result = result[:len(result) // batch_size * batch_size]

    if "search" in skip_tasks:
        for i in range(len(result)):
            result[i]['search_labels'] = None

    if "knowledge" in skip_tasks:
        for i in range(len(result)):
            result[i]['knowledge'] = None

    if "response" in skip_tasks:
        for i in range(len(result)):
            result[i]['response_labels'] = None

    print(f"Num examples: {len(result)}")

    return result


def parse_data(
    data,
    keys,
    passage_count,
    short_replies_len=1
):
    dialog = data['dialog_history']

    context = []

    search_labels = []

    knowledge_passages = []
    knowledge = []

    response_labels = []

    actions_stack = []
    prev_action = '-1'
    prev_context = []
    for idx, elem in enumerate(dialog):
        if not (elem['action'] == keys['user1'] and prev_action == keys['user2']):
            actions_stack.append(elem)
            prev_action = elem['action']
            if idx < len(dialog) - 1:
                continue

        if keys['user2'] not in [i['action'] for i in actions_stack]:
            continue

        prev_action = elem['action']

# Get search label
        do_search = False
        for i in reversed(range(len(actions_stack))):
            if actions_stack[i]['action'] == keys['search']:
                search_labels.append(actions_stack[i]['text'])
                do_search = True
                break

        if not do_search:
            search_labels.append(im.NO_QUERY)

        # Get context and response label
        to_add = True
        response = ''
        prev_token = ''
        count = 0
        context_list = []
        for i in reversed(actions_stack):
            if to_add and i['action'] != keys['user2']:
                to_add = False
                response = im.USER2_REPLY + response[1:]
                context_list[0][0] = im.USER2_REPLY

            if to_add:
                response = '\n' + i['text'] + response
                context_list = [["", i['text']]] + context_list
                count += 1

            if not to_add and i['action'] in [keys['user1'], keys['user2']]:
                token = im.USER1_REPLY if i['action'] == keys['user1'] else im.USER2_REPLY
                # In case when user sends 2+ replies
                # add special token only before first reply
                token = token * (token != prev_token)

                context_list = [[token, i['text']]] + context_list

        if to_add:
            response = im.USER2_REPLY + response[1:]
            context_list[0][0] = im.USER2_REPLY

        # Last user's replies not included in context
        # They are in response
        prev_context.extend(context_list)
        context.append(prev_context[:-count])
        response_labels.append(response)

        # Get knowledge passages and labels
        possible_knowledge = actions_stack[-1]['context']['contents']
        if possible_knowledge != []:
            passages = reduce(
                lambda x, y: x + y,
                [possible_knowledge[i]['content'] for i in range(len(possible_knowledge))]
            )
        else:
            passages = []

        try:
            knowledge_line = reduce(
                lambda x, y: x + y,
                actions_stack[-1]['context']['selected_contents']
            )[1:]
        except KeyError:
            [print(f"{i['action']}: {i['text']}") for i in actions_stack]
            [print(f"{i['action']}: {i['text']}") for i in dialog]

        for i in reversed(range(len(actions_stack) - 1)):
            if actions_stack[i]['action'] == keys['user1']:
                break
            if actions_stack[i]['action'] == keys['user2']:
                current_knowledge = reduce(
                    lambda x, y: x + y,
                    actions_stack[i]['context']['selected_contents']
                )[1:]

                for j in range(len(current_knowledge)):
                    knowledge_line[j] = knowledge_line[j] or current_knowledge[j]

        random.shuffle(passages)
        passages, knowledge_line = reduce_passages_count(passages, knowledge_line, passage_count)

        if len(passages) > 0:
            total_knowledge = reduce(
                lambda x, y: x + y,
                [(im.KNOWLEDGE + passages[i]) * knowledge_line[i] for i in range(len(passages))]
            )
            if total_knowledge == '':
                total_knowledge = im.NO_KNOWLEDGE
        else:
            total_knowledge = im.NO_KNOWLEDGE

        random.shuffle(passages)
        knowledge_passages.append(passages)

        knowledge.append(total_knowledge)

        if len(response_labels[-1].split(" ")) < short_replies_len:
            context.pop(),
            search_labels.pop(),
            knowledge_passages.pop(),
            knowledge.pop(),
            response_labels.pop()

        actions_stack = [elem]

    # Get personas
    if keys['persona1'] in data:
        persona1 = im.PERSONA1 + data[keys['persona1']]
    else:
        persona1 = ''
    if keys['persona2'] is not None and keys['persona2'] in data:
        persona2 = im.PERSONA2 + data[keys['persona2']]
    else:
        persona2 = ''

    personas = [
        persona1 + persona2 for _ in range(len(context))
    ]

    return [
        {
            'context': context[i],
            'search_labels': search_labels[i],
            'knowledge_passages': knowledge_passages[i],
            'knowledge': knowledge[i],
            'personas': personas[i],
            'response_labels': response_labels[i]
        }
        for i in range(len(context))]


def reduce_passages_count(passages, is_used, count):
    result_passages = []
    result_used = []
    total_count = 0
    for i in range(len(passages)):
        if total_count >= count:
            break
        if is_used[i]:
            result_passages.append(passages[i])
            result_used.append(is_used[i])
            total_count += 1
    for i in range(len(passages)):
        if total_count >= count:
            break
        if not is_used[i]:
            result_passages.append(passages[i])
            result_used.append(is_used[i])
            total_count += 1
    return result_passages, result_used


def collate_fn(data, args):
    context = []
    search_labels = []
    knowledge_passages = []
    knowledge = []
    personas = []
    response_labels = []
    for i in data:
        context.append(i['context'])
        search_labels.append(i['search_labels'])
        knowledge_passages.append(i['knowledge_passages'])
        knowledge.append(i['knowledge'])
        personas.append(i['personas'])
        response_labels.append(i['response_labels'])

    last_reply = ""

    if search_labels[0] is None:
        search_ids = None
        search_mask = None
        search_labels = None
    else:
        search_ids, search_mask, search_labels = get_search_data(
            args.tokenizer,
            args.context_max_length,
            args.labels_max_length,
            context,
            personas,
            search_labels
        )

    if knowledge[0] is None:
        knowledge_ids = None
        knowledge_mask = None
        knowledge_labels = None
    else:
        knowledge_ids, knowledge_mask, knowledge_labels = get_knowledge_data(
            args.tokenizer,
            args.passage_max_length,
            args.labels_max_length,
            args.passage_count,
            context,
            knowledge_passages,
            knowledge
        )

    if response_labels[0] is None:
        response_ids = None
        response_mask = None
        response_labels = None
    else:
        response_ids, response_mask, response_labels = get_response_data(
            args.tokenizer,
            args.context_max_length,
            args.labels_max_length,
            context,
            knowledge,
            personas,
            response_labels
        )

    return (
        search_ids, search_mask, search_labels,
        knowledge_ids, knowledge_mask, knowledge_labels,
        response_ids, response_mask, response_labels, last_reply
    )


def get_search_data(
    tokenizer,
    context_max_length,
    labels_max_length,
    context,
    personas,
    search_labels
):
    # Join all replies in one string line
    context = ["\n".join([i[0] + i[1] + p for i in j[-1:]]) for j, p in zip(context, personas)]
    context = [i + im.SEARCH_TASK for i in context]
    LM = tokenizer("<LM>").input_ids   # <LM> prefix only for FRED models

    ids = tokenizer(
        context,
        max_length=context_max_length - 1,
        padding=True,
        truncation=True
    )
    ids, mask = ids.input_ids, ids.attention_mask
    ids = [LM + i for i in ids]
    mask = [[1] + i for i in mask]

    labels = get_labels(search_labels, tokenizer, labels_max_length, im.SEARCH_RESULT)

    ids = torch.LongTensor(ids)
    mask = torch.LongTensor(mask)
    labels = torch.LongTensor(labels)

    return ids, mask, labels


def get_knowledge_data(
    tokenizer,
    passage_max_length,
    labels_max_length,
    passage_count,
    context,
    knowledge_passages,
    knowledge
):
    # Join all replies in one string line
    context = ["\n".join([i[0] + i[1] for i in j]) for j in context]
    LM = tokenizer("<LM>").input_ids   # <LM> prefix only for FRED models

    task_ids = tokenizer.encode(im.KNOWLEDGE_TASK)

    c_ids = tokenizer(
        context,
        max_length=2 * passage_max_length,
        padding=False,
        truncation=True
    )
    c_ids, c_mask = c_ids.input_ids, c_ids.attention_mask
    c_ids = [LM + i for i in c_ids]
    c_mask = [[1] + i for i in c_mask]

    pad = tokenizer.pad_token_id

    k_ids, k_mask = [], []
    for i in range(len(c_ids)):
        if knowledge_passages[i] == []:
            knowledge_passages[i] = ['']
        ids = tokenizer(
            [j for j in knowledge_passages[i]],
            max_length=passage_max_length,
            padding=False,
            truncation=True,
        )
        ids, mask = ids.input_ids, ids.attention_mask

        for j in range(len(ids)):
            task_id = task_ids * (j == (len(ids) - 1))
            mask_id = [1] * len(task_id)
            cut_off_len = max(0, len(c_ids[i]) + len(task_id) - 2 * passage_max_length)
            ids[j] = c_ids[i][:-cut_off_len] + ids[j] + task_id + [
                pad for _ in range(3 * passage_max_length - len(c_ids[i][:-cut_off_len]) - len(ids[j]) - len(task_id))
            ]
            mask[j] = c_mask[i][:-cut_off_len] + mask[j] + mask_id + [
                0 for _ in range(3 * passage_max_length - len(c_mask[i][:-cut_off_len]) - len(mask[j]) - len(mask_id))
            ]

        # Add additional passages so the shape will be B * P * L
        # B - Batch size
        # P - Passage count
        # L - Sequence length
        for j in range(len(ids), passage_count):
            if len(k_ids) > 0:
                batch_idx = random.randint(0, len(k_ids) - 1)
                elem_idx = random.randint(0, len(k_ids[batch_idx]) - 1)

                ids.append(k_ids[batch_idx][elem_idx].copy())
                mask.append(k_mask[batch_idx][elem_idx].copy())
            else:
                elem_idx = random.randint(0, len(ids) - 1)

                ids.append(ids[elem_idx])
                mask.append(mask[elem_idx])

        k_ids.append(ids)
        k_mask.append(mask)

    labels = get_labels(knowledge, tokenizer, labels_max_length, im.KNOWLEDGE_RESULT)

    k_ids = torch.LongTensor(k_ids)
    k_mask = torch.LongTensor(k_mask)
    labels = torch.LongTensor(labels)

    return k_ids, k_mask, labels


def get_response_data(
    tokenizer,
    context_max_length,
    labels_max_length,
    context,
    knowledge,
    personas,
    response_labels
):
    context = ["\n".join([i[0] + i[1] for i in j]) for j in context]

    if knowledge[0] is None:
        knowledge = [im.NO_KNOWLEDGE]

    ids_list = []
    mask_list = []

    for c, p, k in zip(context, personas, knowledge):
        # <LM> prefix only for FRED models
        full_context = "<LM>" + c + p + k + im.RESPONSE_TASK

        ids = tokenizer(
            full_context,
            max_length=context_max_length - 1,
            padding="max_length",
            truncation=True
        )

        ids, mask = ids.input_ids, ids.attention_mask

        ids_list.append(ids)
        mask_list.append(mask)

    labels = get_labels(response_labels, tokenizer, labels_max_length, im.RESPONSE_RESULT)

    ids = torch.LongTensor(ids_list)
    mask = torch.LongTensor(mask_list)
    labels = torch.LongTensor(labels)

    return ids, mask, labels


def get_labels(tokens, tokenizer, max_length, special_token):
    labels = tokenizer(
        [special_token + i for i in tokens],
        max_length=max_length - 1,
        padding=False,
        truncation=True
    ).input_ids

    for i in range(len(labels)):
        labels[i].extend([tokenizer.eos_token_id])
        labels[i].extend(-100 for _ in range(max_length - len(labels[i])))

    return labels


class BatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        data_len = len(self.data_source)
        indices = range(data_len)
        b_indices = torch.randperm(math.ceil(data_len / self.batch_size)).tolist()
        return iter([indices[i * self.batch_size: min(data_len, (i + 1) * self.batch_size)] for i in b_indices])
