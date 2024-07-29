import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from hate_speech.ngram_attention_model import NGramAttention
from hate_speech.data_module import set_ngram_dataset


def main(data_path, save, bert_ckpt_path, ngram_ckpt_path):
    bert = BertForSequenceClassification.from_pretrained('cointegrated/rubert-tiny2', num_labels=2).to("cuda")
    bert_ckpt = torch.load(bert_ckpt_path)
    tokenizer = BertTokenizer.from_pretrained('cointegrated/rubert-tiny2')
    bert.load_state_dict(bert_ckpt)

    ngram = NGramAttention()
    ngram_ckpt = torch.load(ngram_ckpt_path)
    ngram.load_state_dict(ngram_ckpt)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Creating dataset for BERT model
    dataset = pd.read_csv(data_path)

    if 'label' in dataset.columns:
        with_labels = True
    else:
        with_labels = False
        dataset['label'] = [-1 for i in range(len(dataset))]
    # Creatinf TensorDataset for ngram_attention model
    ngram_dataset = set_ngram_dataset(dataset)

    test_params = {
        'batch_size': 1,
        'shuffle': False,
        'drop_last': True,
        'num_workers': 0
    }

    testing_loader = DataLoader(ngram_dataset, **test_params)

    # Compute ngram_attention confidences for the dataset
    ngram.to(device)
    ngram.eval()
    results_ngram_test = []
    for _, data in enumerate(tqdm(testing_loader), 0):
        sentences = data[0] if len(data) > 1 else data
        # Preprocessing
        sentences = ngram.preprocess(sentences)
        with torch.no_grad():
            outputs = ngram(sentences.to(device, dtype=torch.float))
            results_ngram_test += outputs

    # Compute bert confidences for the dataset
    results_bert_test = []
    bert.to(device='cpu')
    bert.eval()
    for comment in tqdm(dataset['comment']):
        input_ids = torch.tensor(tokenizer.encode(comment), device='cpu').unsqueeze(0)
        outputs = bert(input_ids)
        results_bert_test.append(outputs.logits)

    alpha = 0.8
    preds = []
    for i, pair in enumerate(zip(results_bert_test, results_ngram_test)):
        pair0 = torch.sigmoid(pair[0].to(device).squeeze(0))
        pair1 = torch.sigmoid(pair[1])
        pred = pair0 * alpha + pair1 * (1 - alpha)
        preds.append(int(pred.argmax()))
    if save is not None:
        with open(save, 'w') as f:
            f.write('comment,prediction\n')
            for i in range(len(preds)):
                f.write(f'{dataset.iloc[i]["comment"]},{(preds[i])}\n')
    if with_labels:
        ans_test = torch.tensor(dataset['label'].values[:len(results_ngram_test)])
        ans = [int(t) for t in ans_test[:len(preds)]]
        f1 = f1_score(ans, preds)
        acc = accuracy_score(preds, ans)
        print("F1:", f1)
        print("Acc:", acc)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to data')
    parser.add_argument('--save',
                        type=str,
                        help='Path to file where final predicton will be stored. If None, confidences are not saved (default: None)')
    parser.add_argument('--bert_ckpt_path', type=str, help='Checkpoint path for BERT model. (default: BERT_CKPT_PATH from config section)')
    parser.add_argument('--ngram_ckpt_path',
                        type=str,
                        help='Checkpoint path for ngram_attention model (default: NGRAM_CKPT_PATH from config section)')
    args = parser.parse_args()
    main(data_path=args.data_path, save=args.save, bert_ckpt_path=args.bert_ckpt_path, ngram_ckpt_path=args.ngram_ckpt_path)
