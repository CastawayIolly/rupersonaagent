import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TrainingArguments, DataCollatorForSeq2Seq, Trainer


def generate_prompt(data_point):
    promt = f"""<s>system
{data_point['system']}</s><s>user
{data_point['user']}</s><s>bot
{data_point['bot']}</s>"""
    return promt


def tokenize(prompt, tokenizer, CUTOFF_LEN=3584, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < CUTOFF_LEN and add_eos_token
    ):

        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(data_point, tokenizer):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt, tokenizer)
    return tokenized_full_prompt


def train(model_name, dataset):
    model_name = "IlyaGusev/saiga2_7b"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = PeftConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        model_name,
        torch_dtype=torch.float16,
        is_trainable=True,
    ).to(device)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    generation_config = GenerationConfig.from_pretrained(model_name)
    print(generation_config)
    data = load_dataset(
        "json",
        data_files={
            'train': 'train/train.json',
            'validation': 'train/val.json'
        }
    )

    train_data = (
        data["train"].map(generate_and_tokenize_prompt, tokenizer)
    )

    val_data = (
        data["validation"].map(generate_and_tokenize_prompt, tokenizer)
    )
    BATCH_SIZE = 4
    MICRO_BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    LEARNING_RATE = 2e-4
    TRAIN_STEPS = 100
    OUTPUT_DIR = "new_model"

    training_arguments = TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_steps=TRAIN_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=False,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=10,
        save_steps=10,
        output_dir=OUTPUT_DIR,
        save_total_limit=10,
        load_best_model_at_end=True,
        report_to=None,
        overwrite_output_dir=True,
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_arguments,
        data_collator=data_collator
    )
    model = torch.compile(model)
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
