import pandas as pd
import numpy as np
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import classification_report, f1_score
from hate_speech.data_module import CustomDataset


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds)
    return {'F1': f1}


def seed_all(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


seed_all(200)


def main(train_data_path, test_data_path, save_ckpt):
    model = BertForSequenceClassification.from_pretrained('cointegrated/rubert-tiny2', num_labels=2).to("cuda")
    tokenizer = BertTokenizer.from_pretrained('cointegrated/rubert-tiny2')

    # Create the dataset and dataloader for the neural network
    train_dataset = pd.read_csv(train_data_path)
    test_dataset = pd.read_csv(test_data_path)
    MAX_LEN = max([len(comment.strip().split(' ')) for comment in train_dataset['comment']])

    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = CustomDataset(train_dataset, tokenizer=tokenizer, max_len=MAX_LEN)
    testing_set = CustomDataset(test_dataset, tokenizer=tokenizer, max_len=MAX_LEN)

    # Initialize trainer
    training_args = TrainingArguments(
        output_dir='hate_speech/bert_results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        logging_dir='hate_speech/bert_logs',
        load_best_model_at_end=True,
        learning_rate=1e-5,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        seed=200)

    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      train_dataset=training_set,
                      eval_dataset=testing_set,
                      compute_metrics=compute_metrics)

    def get_prediction():
        test_pred = trainer.predict(testing_set)
        labels = np.argmax(test_pred.predictions, axis=-1)
        return labels

    # Train model
    trainer.train()
    # Save checkpoint
    torch.save(model.state_dict(), save_ckpt)

    # Save artefacts
    model_path = "hate_speech/fine-tune-bert"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # Get prediction and score
    pred = get_prediction()

    print(classification_report(testing_set.targets, pred))
    print(f1_score(testing_set.targets, pred))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, help='Path to training data')
    parser.add_argument('--test_data_path', type=str, help='Path to testing data')
    parser.add_argument('--save_ckpt', type=str, help='Where to store checkpoints obtained during training')
    args = parser.parse_args()
    main(train_data_path=args.train_data_path, test_data_path=args.test_data_path, save_ckpt=args.save_ckpt)
