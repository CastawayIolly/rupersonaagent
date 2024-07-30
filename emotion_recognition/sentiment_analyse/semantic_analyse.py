import numpy as np
import torch
from datasets import load_metric
from transformers import Trainer, TrainingArguments


def tokenize_function(example, tokenizers):
    encoded_inputs = []
    for tokenizer in tokenizers:
        encoded_inputs.append(tokenizer(example["text"], truncation=True, padding="max_length", return_tensors="pt"))
    return encoded_inputs


def data_collator(batch, tokenizers):
    input_ids = []
    attention_mask = []
    labels = []

    for example in batch:
        for i in range(len(tokenizers)):
            input_ids.append(example[i]["input_ids"].squeeze())
            attention_mask.append(example[i]["attention_mask"].squeeze())
        labels.append(example["label"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.tensor(labels),
    }


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = load_metric("accuracy")

    acc_metric = accuracy.compute(predictions=predictions, references=labels)
    precision = load_metric("precision")
    recall = load_metric("recall")
    f1 = load_metric("f1")

    precision_metric = precision.compute(predictions=predictions, references=labels)
    recall_metric = recall.compute(predictions=predictions, references=labels)
    f1_metric = f1.compute(predictions=predictions, references=labels)

    return {
        "accuracy": acc_metric['accuracy'],
        "precision": precision_metric['precision'],
        "recall": recall_metric['recall'],
        "f1": f1_metric['f1'],
    }


def get_training_args(output_dir, learning_rate, num_train_epochs):
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )


def model_train(
        model,
        training_args,
        train_dataset,
        validation_dataset,
        tokenizer,
        data_collator,
        compute_metrics
):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
