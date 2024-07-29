## Data format

The expected data format is two columns query and candidate, where query contains the dialogue history and candidate contains the next turn of dialogue.

Entry example:
```

DatasetDict({
    train: Dataset({
        features: ['query', 'candidate'],
        num_rows: 209297
    })
    val: Dataset({
        features: ['query', 'candidate'],
        num_rows: 1000
    })
    test: Dataset({
        features: ['query', 'candidate'],
        num_rows: 1000
    })
})
```

```
{
    'query':	'dialog_history',
    'candidate': 'next_turn'
}

{   
    'query': [
        '<p-2> Привет) расскажи о себе', 
        '<p-2> Привет) расскажи о себе <p-1> Привет) под вкусный кофеек настроение поболтать появилось', 
        '<p-2> Привет) расскажи о себе <p-1> Привет) под вкусный кофеек настроение поболтать появилось <p-2> Что читаешь? Мне нравится классика'
        ],
    'candidate': [
        '<p-1> Привет) под вкусный кофеек настроение поболтать появилось', 
        '<p-2> Что читаешь? Мне нравится классика',
        '<p-2> Я тоже люблю пообщаться'
        ]
}
```

For our experiments we used ['Toloka Persona Chat Rus'](https://toloka.ai/ru/datasets/?datasets-category=nlp#datasets).
Default model ['rubert-tiny2'](https://huggingface.co/cointegrated/rubert-tiny2) was used as retriever.
LLM model - ['saiga_mistral_7b_gguf'](https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf/tree/main) 

Data preprocessing was done using ['data_processing.ipynb'](data_processing.ipynb)

## Train Retriever

[`train_biencoder.py`](train_biencoder.py) provides the code to train a model. An example usage of the script is given below:

```shell
python src/rain_biencoder.py \
        --data_path data/toloka_data \
        --model_name "cointegrated/rubert-tiny2" \
        --max_epochs 3 \
        --devices 1 \
        --save_path "bi_encoder/biencoder_checkpoint.ckpt" \
```

## Test Retriever

You can evaluate your model or a pretrained model with [`test_biencoder.py`](test_biencoder.py). An example usage of the script is provided below.

```shell
python src/biencoder_test.py \
        --model_name
        --checkpoint_dir bi_encoder/biencoder_checkpoint.ckpt \
        --data_path data/toloka_data \
```

## Sample bot interaction

You can try chatting with your model using [`demo.py`](demo.py)

```shell
gradio demo.py 
```