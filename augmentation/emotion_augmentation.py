from transformers import AutoTokenizer, AutoModelForCausalLM
from langdetect import detect


def generate_prompt(source_sentence, emotion=None):
    if detect(source_sentence) == 'ru':
        if emotion is None:
            return f'Перефразируй следующее предложение так, ' \
                   f'чтобы сохранить его основную эмоцию и смысл: {source_sentence}. ' \
                   f'Убедись, что стиль и тональность остаются неизменными.'
        else:
            return f'Перефразируй следующее предложение, добавив в него {emotion}: ' \
                   f'{source_sentence}. Убедись, что новая версия сохраняет основной смысл, ' \
                   f'но передает указанную эмоцию более ярко.'
    elif detect(source_sentence) == 'en':
        if emotion is None:
            return f'Rephrase the following sentence while maintaining the original emotion' \
                   f' and tone: {source_sentence}'
        else:
            return f'Rephrase the following sentence while incorporating {emotion} into the tone ' \
                   f'and wording. Here’s the sentence: {source_sentence}. ' \
                   f'Make sure the new version effectively conveys the specified emotion'


def paraphrase(prompt, max_tokens, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    encoded_input = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda:0')
    output = model.generate(
        **encoded_input,
        num_beams=2,
        do_sample=True,
        max_new_tokens=max_tokens
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)
