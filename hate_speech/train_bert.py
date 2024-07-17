import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
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

def main():
    model = BertForSequenceClassification.from_pretrained('cointegrated/rubert-tiny2', num_labels=2).to("cuda")
    tokenizer = BertTokenizer.from_pretrained('cointegrated/rubert-tiny2')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    # Create the dataset and dataloader for the neural network
    df = pd.read_csv("hate_speech/out_data/ToxicRussianComments.csv")
    MAX_LEN = max([len(comment.strip().split(' ')) for comment in df['comment']])
    train_size = 0.8
    train_dataset=df.sample(frac=train_size,random_state=200)
    test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)


    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = CustomDataset(train_dataset, tokenizer=tokenizer, max_len=MAX_LEN)
    testing_set = CustomDataset(test_dataset, tokenizer=tokenizer, max_len=MAX_LEN) 

    # Initialize trainer
    training_args = TrainingArguments(
                    output_dir = 'hate_speech/bert_results',
                    num_train_epochs = 3,
                    per_device_train_batch_size = 8,
                    per_device_eval_batch_size = 8,
                    weight_decay =0.01,
                    logging_dir = 'hate_speech/bert_logs',
                    load_best_model_at_end = True,
                    learning_rate = 1e-5,
                    evaluation_strategy ='epoch',
                    logging_strategy = 'epoch',
                    save_strategy = 'epoch',
                    save_total_limit = 1,
                    seed=200)
    
    trainer = Trainer(model=model,
                  tokenizer = tokenizer,
                  args = training_args,
                  train_dataset = training_set,
                  eval_dataset = testing_set,
                  compute_metrics = compute_metrics)
    
    def get_prediction():
        test_pred = trainer.predict(testing_set)
        labels = np.argmax(test_pred.predictions, axis = -1)
        return labels
    
    # Train model
    trainer.train()
    # Save checkpoint
    torch.save(model.state_dict(), 'hate_speech/bert_ckpt.pt')

    # Save artefacts
    model_path = "hate_speech/fine-tune-bert"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # Get prediction and score
    pred = get_prediction()
    
    print(classification_report(testing_set.targets, pred))
    print(f1_score(testing_set.targets, pred))

if __name__ == "__main__":
    main()    