from ling_rules import count_features
import json
import pymorphy2
import spacy
from utils import (
    assign_lexical_items,
    count_metrics,
    combine_features,
)

nlp = spacy.load("ru_core_news_lg")
morph = pymorphy2.MorphAnalyzer()


data_path = "data/TolokaPersonaChat_gk_1_500_withoutgender.jsonl"


# get reference text
result_human, result_human_text = [], []

with open(data_path, "r", encoding="utf-8") as inp:
    for line in inp:
        line = json.loads(line)
        dialog = line["dialog"]
        for turn_ in dialog:
            turn = turn_["gk"]
            turn1 = turn_["text"]
            turn = "yes" if len(turn) != 0 else "no"
            result_human.append(turn)
            result_human_text.append(turn1)

# features
# dialog replicas
(
    npro,
    nounroot,
    obl,
    xcomp,
    csubj,
    nmod,
    noun_case,
    adj_noun,
    adj,
    prtf,
    pobj,
    verb_plus_inf,
    verb_constr,
    npro_descripive,
    verb_plus_obj,
    inf_plus_obj,
    prtf_noun,
    digit_noun,
    numr,
    cc_conj,
) = count_features(data_path, persona_description=False)


(
    npro_p,
    nounroot_p,
    obl_p,
    xcomp_p,
    csubj_p,
    nmod_p,
    noun_case_p,
    adj_noun_p,
    adj_p,
    prtf_p,
    pobj_p,
    verb_plus_inf_p,
    verb_constr_p,
    npro_descripive_p,
    verb_plus_obj_p,
    inf_plus_obj_p,
    prtf_noun_p,
    digit_noun_p,
    numr_p,
    cc_conj_p,
) = count_features(data_path, persona_description=True)


# assign features with each sentence in dataframe
cc_conj_tag = assign_lexical_items(result_human_text, cc_conj, cc_conj_p)
adj_tag = assign_lexical_items(result_human_text, adj, adj_p)
# other:
# npro_tag, nounroot_tag, \
# obl_tag, xcomp_tag, \
# csubj_tag, nmod_tag, \
# noun_case_tag, adj_noun_tag, \
# adj_tag, prtf_tag, pobj_tag,verb_plus_inf_tag, verb_constr_tag, \
# npro_descripive_tag, verb_plus_obj_tag, inf_plus_obj_tag, \
# prtf_noun_tag, digit_noun_tag, \
# numr_tag, cc_conj_tag


# count metrics for each feature
metrics_cc_conj = count_metrics(result_human, cc_conj_tag)
# etc


# combine features with the highest f1, acuracy scores
example = combine_features(cc_conj_tag, adj_tag)

metrics_cc_conj_adj = count_metrics(result_human, example)
