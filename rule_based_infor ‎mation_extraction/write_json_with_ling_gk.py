import spacy
from nltk.stem.snowball import SnowballStemmer
import json
import pymorphy2
from ling_rules import (
    get_lexeme,
    get_morph_tags,
    nounroot,
    verb_inf,
    verb_obj,
    npro_descr,
    verb_constr,
)

nlp = spacy.load("ru_core_news_lg")
stemmer = SnowballStemmer("russian")
morph = pymorphy2.MorphAnalyzer()

# lexical constructions with stemming/ without stemming
# with stemming by default
data_path = "data/TolokaPersonaChat_gk_1_500_withoutgender.jsonl"
alp = list("qwertyuiopasdfghjklzxcvbnm")
L = {}
# ngram=[]
with open("data/new.jsonl", "w", encoding="utf-8") as out:
    with open(data_path, "r", encoding="utf-8") as inp:
        for line in inp:
            line = json.loads(line)
            dialog = line["dialog"]
            for turn_i, turn_ in enumerate(dialog):
                doc = nlp(turn_["text"])
                dependences = []
                ngram = []
                for token in doc:
                    dep = token.dep_
                    dependences.append(dep)
                    turn = turn_["text"].split(" ")
                    syntax = list(zip(turn, dependences))
                    for lex_dep, lex_dep2 in zip(syntax, syntax[1:]):
                        # word,word_next=get_lexeme(lex_dep, lex_dep2)
                        sintagma, sintagma_next = lex_dep[1], lex_dep2[1]
                        word_parse, word_parse2 = get_morph_tags(
                            lex_dep, lex_dep2)
                        # complex rules
                        is_nounroot = nounroot(word_parse, sintagma)
                        is_verb_inf = verb_inf(word_parse, sintagma_next)
                        is_verb_obj = verb_obj(word_parse, sintagma_next)
                        is_npro_descr = npro_descr(word_parse, word_parse2)
                        is_verb_constr = verb_constr(word_parse, word_parse2)

                        if word_parse.tag.POS == "NOUN":
                            ngram.append(word_parse.normal_form)
                        elif word_parse.tag.POS == "INFN":
                            ngram.append(word_parse.normal_form)
                        elif is_verb_constr:
                            ngram.append(word_parse2.normal_form)
                        elif is_verb_obj:
                            ngram.append(word_parse.normal_form)
                            ngram.append(word_parse2.normal_form)
                        elif is_verb_inf:
                            ngram.append(word_parse.normal_form)
                            ngram.append(word_parse2.normal_form)
                        elif sintagma == "xcomp" and sintagma_next == "obj":
                            ngram.append(word_parse.normal_form)
                            ngram.append(word_parse2.normal_form)
                        elif sintagma == "obl" or sintagma == "nmod":
                            ngram.append(word_parse.normal_form)
                        elif word_parse.tag.POS in ["NOUN", "ADJF", "PRTF"]:
                            ngram.append(word_parse.normal_form)
                        elif (
                            word_parse.tag.POS == "ADJF" or word_parse.tag.POS == "PRTF"
                        ) and word_parse2.tag.POS == "NOUN":
                            ngram.append(word_parse.normal_form)
                            ngram.append(word_parse2.normal_form)
                        elif word_parse.tag.POS == "NUMR":
                            ngram.append(word_parse.normal_form)
                            ngram.append(word_parse2.normal_form)
                        elif is_npro_descr:
                            ngram.append(word_parse2.normal_form)

                    turn = ngram
                    # WITH STEMS
                    # ngram=[stemmer.stem(word) for word in ngram]
                    persona_ = line["persons"][turn_["person"]]
                    persona = []
                    for p in persona_:
                        doc_p = nlp(p)
                        dependences_p = []
                        ngrams = []
                        p = p.split()
                        for token in doc_p:
                            dep = token.dep_
                            dependences_p.append(dep)
                            syntax = list(zip(p, dependences_p))
                            for lex_dep, lex_dep2 in zip(syntax, syntax[1:]):
                                word, word_next = get_lexeme(lex_dep, lex_dep2)
                                sintagma, sintagma_next = lex_dep[1], lex_dep2[1]
                                word_parse, word_parse2 = get_morph_tags(
                                    lex_dep, lex_dep2
                                )
                                # complex rules
                                is_nounroot = nounroot(word_parse, sintagma)
                                is_verb_inf = verb_inf(
                                    word_parse, sintagma_next)
                                is_verb_obj = verb_obj(
                                    word_parse, sintagma_next)
                                is_npro_descr = npro_descr(
                                    word_parse, word_parse2)
                                is_verb_constr = verb_constr(
                                    word_parse, word_parse2)

                                if is_nounroot:
                                    ngrams.append(word_parse.normal_form)
                                elif word_parse.tag.POS == "INFN":
                                    ngrams.append(word_parse.normal_form)
                                elif is_verb_inf:
                                    ngrams.append(word_parse.normal_form)
                                    ngrams.append(word_parse2.normal_form)
                                elif is_verb_obj:
                                    ngrams.append(word_parse.normal_form)
                                    ngrams.append(word_parse2.normal_form)
                                elif sintagma == "xcomp" and sintagma_next == "obj":
                                    ngrams.append(word_parse.normal_form)
                                    ngrams.append(word_parse2.normal_form)
                                elif (
                                    word_parse.tag.POS == "NOUN" and word_parse.tag.case != "nomn"
                                ):
                                    ngrams.append(word_parse.normal_form)
                                elif sintagma in ["obl", "nmod"]:
                                    ngrams.append(word_parse.normal_form)
                                elif is_npro_descr:
                                    ngrams.append(word_parse2.normal_form)

                                elif word_parse.tag.POS in ["NOUN", "ADJF", "PRTF"]:
                                    ngrams.append(word_parse.normal_form)
                                elif word_parse.tag.POS == "NUMR":
                                    ngrams.append(word_parse.normal_form)
                                    ngrams.append(word_parse2.normal_form)
                                elif word == "[0-9]*":
                                    ngrams.append(word_parse.normal_form)
                                    ngrams.append(word_parse2.normal_form)

                                elif (word_parse.tag.POS == "ADJF" or word_parse.tag.POS == "PRTF") and word_parse2.tag.POS == "NOUN":
                                    ngrams.append(word_parse.normal_form)
                                    ngrams.append(word_parse2.normal_form)
                        persona.append(ngrams)
                        # WITH STEMS
                        # persona=[stemmer.stem(word) for word in ngrams]

                    golden_p = set()
                    for i, p in enumerate(persona):
                        for wp in p:
                            for wr in ngram:
                                if wr in wp or wp in wr:
                                    if i not in golden_p:
                                        golden_p.add(i)

                    if len(L) - 1 < len(golden_p):
                        L[len(golden_p)] = 1
                    else:
                        L[len(golden_p)] += 1
                    dialog[turn_i]["gk"] = list(golden_p)

            line = json.dumps(line, ensure_ascii=False)
            out.write(line + "\n")
