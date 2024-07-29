from argparse import ArgumentParser

import torch
from RAG_LLM_dialogue.src.model import BiEncoder, CustomDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets import load_from_disk


def evaluate_model(checkpoint_dir, data_path, model_name, max_length, batch_size, device):
    dataset = load_from_disk(dataset_path=data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_dataset = CustomDataset(dataset['test'], tokenizer, max_length=max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = BiEncoder.load_from_checkpoint(checkpoint_dir)
    model.to(device)
    model.eval()

    mrr, r1, r5 = 0, 0, 0
    total = 0

    with torch.no_grad():
        for batch in test_dataloader:
            query_input_ids = batch['query_input_ids'].to(device)
            query_attention_mask = batch['query_attention_mask'].to(device)
            candidate_input_ids = batch['candidate_input_ids'].to(device)
            candidate_attention_mask = batch['candidate_attention_mask'].to(device)
            query_embeddings, candidate_embeddings = model(
                query_input_ids, query_attention_mask, candidate_input_ids, candidate_attention_mask
            )
            scores = torch.matmul(query_embeddings, candidate_embeddings.T)
            labels = torch.arange(scores.size(0), device=device)
            total += scores.size(0)
            for i in range(scores.size(0)):
                sorted_indices = torch.argsort(scores[i], descending=True)
                rank = (sorted_indices == labels[i]).nonzero(as_tuple=True)[0].item() + 1
                mrr += 1.0 / rank
                if rank == 1:
                    r1 += 1
                if rank <= 5:
                    r5 += 1
    mrr /= total
    r1 /= total
    r5 /= total
    print(f'MRR: {mrr}')
    print(f'Recall@1: {r1}')
    print(f'Recall@5: {r5}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--model_name', type=str, default='cointegrated/rubert-tiny2', help='Model name or path')
    parser.add_argument('--max_length', type=int, default=64, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda or cpu)')
    args = parser.parse_args()
    evaluate_model(args.checkpoint_dir, args.data_path, args.model_name, args.max_length, args.batch_size, args.device)
