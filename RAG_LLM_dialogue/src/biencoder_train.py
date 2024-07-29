from argparse import ArgumentParser

from datasets import load_from_disk
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from RAG_LLM_dialogue.src.model import BiEncoder, CustomDataset

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='cointegrated/rubert-tiny2')
    parser.add_argument('--data_path', type=str, required=True, help='Path to your data')
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='bi_encoder/biencoder_checkpoint.ckpt', help='Path to save the model checkpoint')
    args = parser.parse_args()

    dataset = load_from_disk(dataset_path=args.data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = CustomDataset(dataset['train'], tokenizer, max_length=args.max_length)
    val_dataset = CustomDataset(dataset['val'], tokenizer, max_length=args.max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = BiEncoder(args.model_name, lr=args.lr)
    trainer = Trainer(accelerator='gpu', devices=args.devices, max_epochs=args.max_epochs)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.save_checkpoint(args.save_path)
