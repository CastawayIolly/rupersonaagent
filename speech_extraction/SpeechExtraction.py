import pandas as pd
from speech_extraction.SpeechCharacteristic import (
    SpeechCharacteristic,
    characteristics
)
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from statistics import mean
import string
import operator
import functools
from collections import Counter
import scipy.stats as stats

nltk.download('punkt')


def read_dataset(
        dataset_name: str,
        path: str,
        column_name: str,
        delimiter: str):
    df = pd.read_csv(path, delimiter=delimiter)
    return df[column_name], dataset_name


def get_sentence_and_words(dataset):
    sentences = []
    words = []
    letters_in_words = []
    punctuations = []
    letters = string.ascii_letters
    punctuation = string.punctuation
    for key, value in dataset.iteritems():
        sentences.append(len(sent_tokenize(value)))
        for sent in sent_tokenize(value):
            words_count = len(word_tokenize(sent))
            words.append(words_count)
            letter_count = len(
                list(
                    filter(functools.partial(
                        operator.contains,
                        letters),
                        sent)
                )
            )
            punctuation_count = len(
                list(
                    filter(
                        functools.partial(
                            operator.contains,
                            punctuation
                        ),
                        sent
                    )
                )
            )
            letters_in_words.append(letter_count / words_count)
            punctuations.append(punctuation_count)
    return (
        mean(sentences),
        mean(words),
        mean(letters_in_words),
        mean(punctuations),
        sum(sentences),
    )


def get_pos_tags(dataset, size):
    text = ' '.join([v for k, v in dataset.iteritems()])
    tokens = nltk.word_tokenize(text.lower())
    new_text = nltk.Text(tokens)
    tags = nltk.pos_tag(new_text)
    counts = Counter(tag for word, tag in tags)
    return dict((word, float(count) / size) for word, count in counts.items())


def get_speech_characteristic(dataset, dataset_name: str):
    speech = SpeechCharacteristic(dataset_name=dataset_name)
    (sentences, words, letters,
     punctuation, size) = get_sentence_and_words(dataset)
    speech.sentences_in_speech = sentences
    speech.words_in_phrase = words
    speech.letter_in_words = letters
    speech.punctuation = punctuation
    pos_tags = get_pos_tags(dataset, size)

    for ch in characteristics:
        if pos_tags.get(ch):
            speech.__dict__[ch] = pos_tags[ch]

    return speech


def get_info_from_dataset(
    dataset_name: str, path: str, column_name: str, delimiter: str
):
    dataset, dataset_name = read_dataset(
        dataset_name,
        path,
        column_name,
        delimiter
    )
    speech_characteristic = get_speech_characteristic(dataset, dataset_name)
    return speech_characteristic


def get_info_from_sentence(sentence: str, column_name: str, phrase_name: str):
    dataset = pd.Series({column_name: sentence})
    speech_characteristic = get_speech_characteristic(dataset, phrase_name)
    return speech_characteristic


def compare_characteristics(
    first_characteristic: SpeechCharacteristic,
    second_characteristic: SpeechCharacteristic,
):
    first = list(first_characteristic.__dict__.values())[1:]
    second = list(second_characteristic.__dict__.values())[1:]
    _, pnorm = stats.mannwhitneyu(first, second, use_continuity=False)
    return pnorm
