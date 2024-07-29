# About

This repository adapts the original FiD code (<https://github.com/facebookresearch/FiD>) to work for personolized dialogue generation. The code was changed to fit the new task where needed. 

## Dependencies

- Python 3
- [PyTorch](http://pytorch.org/) (currently tested on version 1.6.0)
- [Transformers](http://huggingface.co/transformers/) (**version 3.0.2**, unlikely to work with a different version)
- [NumPy](http://www.numpy.org/)


## Data format

The expected data format is a list of entry examples, where each entry example is a dictionary in the following format

Entry example:
```
{ 
        'id': '0', 
        'question': 'Last utterance', 
        'target': 'Golden reply', 
        'answers': ['same as target'], 
        'ctxs': 
        [ 
                { 
                        "title": "Persona_train_1", 
                        "text": "dialogue history" 
                }, 
                { 
                        "title": "Persona_train_2", 
                        "text": "dialogue history”
                }
        ] 
}
```

Since a dialog consists of consecutive utterances, for a dialog where the bot has n utterances, we will need to create n such entries.


For our experiments we used ['Toloka Persona Chat Rus'](https://toloka.ai/ru/datasets/?datasets-category=nlp#datasets).

Data preprocessing was done using ['persona_chat_preprocess.ipynb'](persona_chat_preprocess.ipynb)

# I. Fusion-in-Decoder

Fusion-in-Decoder models can be trained using [`train_reader.py`](reader_train.py) and evaluated with [`test_reader.py`](reader_test.py).

all the script parameters can be viewed in the [`options.py`](src/options.py) file.

## Train

[`train_reader.py`](reader_train.py) provides the code to train a model. An example usage of the script is given below:

```shell
python reader_train.py \
        --train_data train_data.json \
        --eval_data eval_data.json \
        --base_model_path "path to your T5 model" \
        --per_gpu_batch_size 1 \
        --name my_experiment \
        --checkpoint_dir checkpoint \
```

Tensors of variable sizes lead to memory overhead. Encoder input tensors have a fixed size by default, but not the decoder input tensors. The tensor size on the decoder side can be fixed using `--answer_maxlength`.

## Test

You can evaluate your model or a pretrained model with [`test_reader.py`](reader_test.py). An example usage of the script is provided below.

```shell
python reader_test.py \
        --base_model_path
        --model_path checkpoint_dir/my_experiment/my_model_dir/checkpoint/best_dev \
        --eval_data eval_data.json \
        --per_gpu_batch_size 1 \
        --name my_test \
        --checkpoint_dir checkpoint \
```

## Sample bot interaction

fidagent.py contains a simple class that wraps a fine-tuned model.
You can try chatting with your model following the instructions provided in dialogue_agent_test.ipynb