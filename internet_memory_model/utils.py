import os

import nest_asyncio
from search.engines.yahoo import Search as YahooSearch
from sklearn.metrics import f1_score
import numpy as np
from bs4 import BeautifulSoup
from functools import reduce
import requests
import urllib.parse


def log_metrics(
    pl_module,
    logs=None,
    log_dicts=None,
    batch_size=1
):
    if logs is not None:
        for log in logs:
            pl_module.log(
                log['name'],
                log['value'],
                on_epoch=log['on_epoch'],
                on_step=log['on_step'],
                batch_size=batch_size
            )
    if log_dicts is not None:
        for log in log_dicts:
            pl_module.log_dict(
                log['value'],
                on_epoch=log['on_epoch'],
                on_step=log['on_step'],
                batch_size=batch_size
            )


def compute_metrics(pl_module, answer, labels, generation=False):
    """
    Method for computing BLEU-1/2 and F1 metrics
    """
    y_pred = [
        pl_module.tokenizer.decode(i, skip_special_tokens=False) if not generation else
        pl_module.tokenizer.decode(i[1:], skip_special_tokens=False)
        for i in answer
    ]
    y_pred = [
        i[:i.find(pl_module.tokenizer.eos_token)]
        for i in y_pred
    ]
    for i in range(len(y_pred)):
        if y_pred[i] == '':
            y_pred[i] = 'no result'

    labels = labels.detach().cpu().numpy()
    y_true = []
    for i in labels:
        temp = []
        for j in i:
            if j != -100:
                temp.append(j)
        result = pl_module.tokenizer.decode(temp, skip_special_tokens=False)
        result = result[:result.find(pl_module.tokenizer.eos_token)]
        if result == '':
            result = 'no result'
        y_true.append(result)

    metrics = pl_module.metrics(y_pred, [y_true])
    y_true = labels[0]
    idx = list(np.where(y_true == -100)[0])
    idx = y_true.shape[0] if idx == [] else idx[0]
    y_true = y_true[:idx]

    y_pred = answer[0].cpu().numpy()
    max_len = max(y_true.shape[0], y_pred.shape[0])
    if y_pred.shape[0] < max_len:
        y_pred = np.concatenate((y_pred, [pl_module.tokenizer.pad_token_id for _ in range(max_len - y_pred.shape[0])]))
    if y_true.shape[0] < max_len:
        y_true = np.concatenate((y_true, [pl_module.tokenizer.pad_token_id for _ in range(max_len - y_true.shape[0])]))
    f1 = f1_score(y_true, y_pred, average='macro')

    return metrics, f1


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class SearchModule():
    """
    Class for Internet search
    """
    def __init__(self, languages=['ru']):
        os.environ["http_proxy"] = "http://proxy.ad.speechpro.com:3128"
        os.environ["https_proxy"] = "http://proxy.ad.speechpro.com:3128"
        os.environ["ftp_proxy"] = "http://proxy.ad.speechpro.com:3128"

        nest_asyncio.apply()

        yahoo = YahooSearch()
        # duckduckgo = DuckDuckSearch()

        self.search_engines = [
            yahoo,
            # duckduckgo
        ]

        self.languages = languages

    def do_search(self, query, page_num=1):
        docs = []
        for i in range(page_num):
            search_args = (query, i)

            docs.extend(self.get_one_page_results(search_args))

        return docs

    def get_one_page_results(self, search_args):
        docs = []
        for i, engine in enumerate(self.search_engines):
            results = None
            try:
                results = engine.search(*search_args, hl='ru')
            except Exception as e:
                print(f"{e}\nSkipping {engine}")

            if results is not None:
                links = [i['links'] for i in results]

            if i == 1:
                # Preprocess link if it's from DuckDuckGo
                for j in links:
                    print(j)
                    o = urllib.parse.urlparse(j)
                    d = urllib.parse.parse_qs(o.query)
                    link = d['uddg'][0] if 'uddg' in d else j

                    docs.extend(self.text_from_html(link))
            else:
                docs.extend(reduce(
                    lambda x, y: x + y,
                    [self.text_from_html(j) for j in links]
                ))

        return docs

    def text_from_html(self, link):
        try:
            body = requests.get(link, timeout=5)
        except Exception:
            return []

        if body.status_code == 200:
            ans = []
            soup = BeautifulSoup(body.text, 'html.parser')

            for i in soup.find_all(['p', 'ul']):
                ans.append(i.get_text())

            ans = [i.replace("\xa0", " ") for i in ans]
            ans = [i.strip(" \n") for i in ans]

            return ans
        else:
            return []
