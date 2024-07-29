import numpy as np
from torch.utils.data import DataLoader
from torch import cuda
from tqdm import tqdm
import torch
from hate_speech.data_module import set_ngram_dataset
from sklearn.metrics import f1_score, accuracy_score
from hate_speech.ngram_attention_model import NGramAttention
# from data_module import CustomDataset, set_ngram_dataset
# from ngram_attention_model import NGramAttention


# Config section
BATCH_SIZE = 5
EPOCHS = 5
LEARNING_RATE = 0.0001


def main(data_path,
         save,
         ckpt_path,
         mode='train',
         total_epochs=EPOCHS,
         batch_size=BATCH_SIZE,
         learning_rate=LEARNING_RATE):

    print(f"data_path: {data_path}")
    print(f"mode: {mode}")

    device = 'cuda' if cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    dataset = set_ngram_dataset(data_path)

    loader_params = {
        'batch_size': batch_size,
        'shuffle': True,
        'drop_last': True,
        'num_workers': 0}

    # Create dataloaders
    data_loader = DataLoader(dataset, **loader_params)

    # Initialize model
    model = NGramAttention()
    model.to(device)

    # Define loss function
    def loss_fn(outputs, targets):
        return torch.nn.CrossEntropyLoss()(outputs, targets)
    # Initialize optimizer and scheduler for further training
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    def train(epoch):
        if 'train' not in data_path.split('/')[-1]:
            print("Make sure train part of the dataset is used for training, not the test part")
        model.train()
        for index, data in enumerate(tqdm(data_loader), 0):
            sentences = data[0]
            targets_ = data[1]
            targets = torch.empty((len(data[1]), 2), dtype=torch.float)
            # Substitute 0 in labels with [1., 0.] and 1 in labels with [0., 1.]
            for i, tar in enumerate(targets_):
                if tar == 0:
                    targets[i] = torch.tensor([1., 0.])
                else:
                    targets[i] = torch.tensor([0., 1.])
            # Preprocessing
            sentences = model.preprocess(sentences)
            # Training loop content
            outputs = model(sentences.to(device, dtype=torch.float))
            optimizer.zero_grad()
            loss = loss_fn(outputs, targets.to(device, dtype=torch.float))
            # Log loss every 50 iterations
            if index % 50 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            loss.backward()
            optimizer.step()
        scheduler.step()
        # Log learning rate when epoch ends
        print("Current LR: ", scheduler.get_last_lr())
        # Save checkpoint after every epoch
        ckpt = model.state_dict()
        ckpt_path = f"{save}/checkpoint_{epoch}.pt"
        torch.save(ckpt, ckpt_path)
        print(f"Epoch {epoch} | Training checkpoint saved at {ckpt_path}")

    def test():
        print(f'ckpt_path: {ckpt_path}')
        if 'test' not in data_path:
            print("Make sure test part of the dataset is used, not the train part")
        print("Loading model from checkpoint...")
        checkpoint = torch.load(f'{ckpt_path}')
        model.load_state_dict(checkpoint)
        print("Start evaluating...")
        results = []
        ans = []
        model.eval()
        for _, data in enumerate(tqdm(data_loader), 0):
            sentences = data[0]
            targets = data[1]
            # Preprocessing
            sentences = model.preprocess(sentences)
            # Evaluation loop content
            outputs = model(sentences.to(device, dtype=torch.float))
            results += outputs
            ans += targets

        # Compute and log metrics
        results = np.array([r.cpu().numpy() for r in results])
        results = [(1 if result[1] > result[0] else 0) for result in results]
        ans = np.array([r.cpu().numpy() for r in ans])
        f1 = f1_score(results, ans, average='weighted')
        acc = accuracy_score(results, ans)
        print(f'len test: {len(results)}\n F1: {f1}\n Accuracy: {acc}\n')
        print("Evaluation ended.")

    if mode == 'train':
        for epoch in range(total_epochs):
            train(epoch)
    if mode == 'eval':
        test()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to data for training/testing')
    parser.add_argument('--save', type=str, help='Where to store checkpoints obtained during training')
    parser.add_argument('--mode',
                        default='train',
                        type=str,
                        help='Wheter to run script for model training or for evaluation (default: train)',
                        choices=['train', 'eval'])
    parser.add_argument('--total_epochs', default=EPOCHS, type=int, help='Total epochs to train the model (default: EPOCHS from config section)')
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='Input batch size (default: BATCH_SIZE from config section)')
    parser.add_argument('--learning_rate',
                        default=LEARNING_RATE,
                        type=float,
                        help='Learning rate for train mode (default: LEARNING_RATE from config section)')
    parser.add_argument('--ckpt_path', type=str, help='Checkpoint path for test and inference (default: CKPT_PATH from config section)')
    args = parser.parse_args()
    main(data_path=args.data_path,
         save=args.save,
         mode=args.mode,
         ckpt_path=args.ckpt_path,
         total_epochs=args.total_epochs,
         batch_size=args.batch_size,
         learning_rate=args.learning_rate)
