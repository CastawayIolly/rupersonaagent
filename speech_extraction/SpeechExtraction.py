import pandas as pd
from speech_extraction.SpeechCharacteristic import SpeechCharacteristic
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from statistics import mean
import string
import operator
import functools
from collections import Counter
import scipy.stats as stats
nltk.download('punkt')


def read_dataset(dataset_name: str, path: str,
                 column_name: str, delimeter: str):
    df = pd.read_csv(path, delimiter=delimeter)
    return df[column_name], dataset_name


def get_sentence_and_words(dataset):
    sentences = []
    words = []
    letters_in_words = []
    punctuations = []
    letters = string.ascii_letters
    punctuation = string.punctuation
    for i, v in dataset.iteritems():
        sentences.append(len(sent_tokenize(v)))
        for sent in sent_tokenize(v):
            words_count = len(word_tokenize(sent))
            words.append(words_count)
            letter_count = len(list(filter(
                functools.partial(operator.contains, letters), sent)))
            punctuation_count = len(list(filter(
                functools.partial(operator.contains,
                                  punctuation), sent)))
            letters_in_words.append(letter_count/words_count)
            punctuations.append(punctuation_count)
    return mean(sentences), mean(words), mean(letters_in_words),\
        mean(punctuations), sum(sentences)


def get_pos_tags(dataset, size):
    text = ''
    for i, v in dataset.iteritems():
        text += v + ' '
    tokens = nltk.word_tokenize(text.lower())
    new_text = nltk.Text(tokens)
    tags = nltk.pos_tag(new_text)
    counts = Counter(tag for word, tag in tags)
    return dict((word, float(count)/size) for word, count in counts.items())


def get_speech_characteristic(dataset, dataset_name: str):
    speech = SpeechCharacteristic(dataset_name=dataset_name)
    sentences, words, \
        letters, punctuation, size = get_sentence_and_words(dataset)
    speech.sentences_in_speech = sentences
    speech.words_in_phrase = words
    speech.letter_in_words = letters
    speech.punctuation = punctuation
    pos_tags = get_pos_tags(dataset, size)
    if pos_tags.get('NN'):
        speech.Noun = pos_tags['NN']
    if pos_tags.get('RB'):
        speech.RB = pos_tags['RB']
    if pos_tags.get('PRP'):
        speech.PRP = pos_tags['PRP']
    if pos_tags.get('VBP'):
        speech.VBP = pos_tags['VBP']
    if pos_tags.get('JJ'):
        speech.JJ = pos_tags['JJ']
    if pos_tags.get('TO'):
        speech.TO = pos_tags['TO']
    if pos_tags.get('VB'):
        speech.VB = pos_tags['VB']
    if pos_tags.get('DT'):
        speech.DT = pos_tags['DT']
    if pos_tags.get('NNS'):
        speech.NNS = pos_tags['NNS']
    if pos_tags.get('IN'):
        speech.IN = pos_tags['IN']
    if pos_tags.get('WRB'):
        speech.WRB = pos_tags['WRB']
    if pos_tags.get('VBD'):
        speech.VBD = pos_tags['VBD']
    if pos_tags.get('VBN'):
        speech.VBN = pos_tags['VBN']
    if pos_tags.get('RP'):
        speech.RP = pos_tags['RP']
    if pos_tags.get('CC'):
        speech.CC = pos_tags['CC']
    if pos_tags.get('VBG'):
        speech.VBG = pos_tags['VBG']
    if pos_tags.get('JJR'):
        speech.JJR = pos_tags['JJR']
    if pos_tags.get('RBR'):
        speech.RBR = pos_tags['RBR']
    if pos_tags.get('WDT'):
        speech.WDT = pos_tags['WDT']
    if pos_tags.get('MD'):
        speech.MD = pos_tags['MD']
    if pos_tags.get('VBZ'):
        speech.VBZ = pos_tags['VBZ']
    if pos_tags.get('WP'):
        speech.WP = pos_tags['WP']
    if pos_tags.get('EX'):
        speech.EX = pos_tags['EX']
    if pos_tags.get('PRP'):
        speech.PRP = pos_tags['PRP']
    if pos_tags.get('CD'):
        speech.CD = pos_tags['CD']
    if pos_tags.get('PDT'):
        speech.PDT = pos_tags['PDT']
    if pos_tags.get('JJS'):
        speech.JJS = pos_tags['JJS']
    if pos_tags.get('POS'):
        speech.POS = pos_tags['POS']
    if pos_tags.get('FW'):
        speech.FW = pos_tags['FW']
    if pos_tags.get('RBS'):
        speech.RBS = pos_tags['RBS']
    if pos_tags.get('NNP'):
        speech.NNP = pos_tags['NNP']
    return speech


def get_info_from_dataset(dataset_name: str, path: str,
                          column_name: str, delimeter: str):
    dataset, dataset_name = read_dataset(dataset_name, path,
                                         column_name, delimeter)
    speech_characteristic = get_speech_characteristic(dataset, dataset_name)
    return speech_characteristic


def get_info_from_sentence(sentence: str, column_name: str, phrase_name: str):
    dataset = pd.Series({column_name: sentence})
    speech_characteristic = get_speech_characteristic(dataset, phrase_name)
    return speech_characteristic


def compare_characteristics(first_characteristic: SpeechCharacteristic,
                            second_characteristic: SpeechCharacteristic):
    first = list(first_characteristic.__dict__.values())[1:]
    second = list(second_characteristic.__dict__.values())[1:]
    _, pnorm = stats.mannwhitneyu(first, second, use_continuity=False)
    return pnorm

r=get_info_from_dataset('us politics', path="/Users/anastasia/Documents/rupersonaagent/data/us.csv", column_name="text", delimeter=",")
v=get_info_from_dataset('hockey', path="/Users/anastasia/Documents/rupersonaagent/data/interview.csv", column_name="text", delimeter=",")

print( compare_characteristics(r,v))