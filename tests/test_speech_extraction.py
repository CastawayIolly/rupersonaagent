import pandas as pd
import pytest
from speech_extraction.SpeechExtraction import (
    read_dataset,
    get_sentence_and_words,
    get_pos_tags,
    get_speech_characteristic,
)
from hamcrest import assert_that, equal_to


@pytest.mark.speech
class TestSpeechExtraction:
    def test_read_dataset(self):
        dataset_value = read_dataset('test_name', 'test_dataset.csv', 'RowId', ',')
        assert_that(dataset_value[1], equal_to("test_name"))
        assert_that(list(dataset_value[0].iteritems())[0], equal_to((0, 0)))

    def test_get_sentence_and_words(self):
        test_data = pd.Series({'text': 'I like cookies'})
        assert_that(get_sentence_and_words(test_data), equal_to((1, 3, 4.0, 0, 1)))

    def test_get_pos_tags(self):
        test_data = pd.Series({'text': 'I like cookies'})
        assert_that(get_pos_tags(test_data, 1), equal_to({'NNS': 2.0, 'IN': 1.0}))

    def test_speech_characteristic(self):
        test_data = pd.Series({'text': 'I like cookies'})
        speech = get_speech_characteristic(test_data, 'test_data')
        assert_that(speech.NNS, equal_to(2.0))
        assert_that(speech.IN, equal_to(1.0))
        assert_that(speech.NNP, equal_to(0.0))
        assert_that(speech.sentences_in_speech, equal_to(1))
        assert_that(speech.words_in_phrase, equal_to(3))
