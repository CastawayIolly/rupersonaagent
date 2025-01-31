# This source code has been adapted from original FiD implementation by
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# The original repository: https://github.com/facebookresearch/FiD
# The original code is licensed by Attribution-NonCommercial 4.0 International (https://creativecommons.org/licenses/by-nc/4.0/)
# This code has been modified to allow for dialogue agents training
# The source code found in this part of the repository is licensed accordingly
# The text of the license can be found in the LICENSE file at the root of this directory

import regex
import string

# Normalization from SQuAD evaluation script
# https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/


def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])
