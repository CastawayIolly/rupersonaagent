import spacy
import json
import pymorphy2

nlp = spacy.load("ru_core_news_lg")
morph = pymorphy2.MorphAnalyzer()

alp = list("qwertyuiopasdfghjklzxcvbnm")


# get current and next lexeme
def get_lexeme(lex_dep, lex_dep2):
    word, word_next = lex_dep[0], lex_dep2[0]
    word = [letter for letter in word
            if letter.isalpha() and letter not in alp]
    word = "".join(word)
    word_next = [
        letter for letter in word_next
        if letter.isalpha() and letter not in alp
    ]
    word_next = "".join(word_next)

    return word, word_next


# get current and next morph tag
def get_morph_tags(lex_dep, lex_dep2):
    word, word_next = get_lexeme(lex_dep, lex_dep2)
    word_parse = morph.parse(word)[0]
    try:
        word_parse2 = morph.parse(word_next)[0]
    except IndexError:
        word_parse2 = "null"
    return word_parse, word_parse2


# nounroot rule
def nounroot(word_parse, sintagma):
    return (word_parse.tag.POS == "NOUN" and sintagma == "ROOT") or \
        (word_parse.tag.POS == "NOUN" and word_parse.tag.animacy == "anim" and word_parse.tag.case == "nomn")


# verb + infinitive construction
def verb_inf(word_parse, sintagma_next):
    return (
        word_parse.tag.POS == "VERB" and word_parse.tag.person == "1per" and word_parse.tag.number == "sing"
    ) and sintagma_next == "xcomp"


# verb + object construction
def verb_obj(word_parse, sintagma_next):
    return (
        word_parse.tag.POS == "VERB" and word_parse.tag.person == "1per" and word_parse.tag.number == "sing"
    ) and sintagma_next == "obj"


# npro + descriptive units
def npro_descr(word_parse, word_parse2):
    return (
        word_parse.tag.POS == "NPRO" and word_parse.tag.person == "1per" and word_parse.tag.number == "sing"
    ) and word_parse2.tag.POS in ["NOUN", "ADJF", "PRTF", "PRTS"]


# verb construction
def verb_constr(word_parse, word_parse2):
    return (
        word_parse.tag.POS == "VERB" and word_parse.tag.person == "1per" and word_parse.tag.number == "sing"
    ) and word_parse2.tag.POS not in ["PREP", "CONJ", "ADVB"]


# count linguistic features
def count_features(data_path, persona_decription=True):
    """
    :data_path: str, jsonl file path
    :persona_decription: bool, 'dialog' or 'persona' data

    """
    npro, nounroot, obl, xcomp, csubj, nmod, \
        noun_case, adj_noun, adj, prtf, pobj = (
            [] for i in range(11)
        )
    verb_plus_inf, verb_constr, npro_descripive = ([] for i in range(3))
    verb_plus_obj, inf_plus_obj, prtf_noun, digit_noun, numr, cc_conj = (
        [] for i in range(6)
    )

    with open(data_path, "r", encoding="utf-8") as inp:
        for line in inp:
            line = json.loads(line)
            dialog = line["dialog"]
            for turn_i, turn_ in enumerate(dialog):
                if persona_decription:
                    persona_ = line["persons"][turn_["person"]]["description"]

                    for p in persona_:
                        doc = nlp(p)
                        dependences_p = []
                        p = p.split()
                        for token in doc:
                            dep = token.dep_
                            dependences_p.append(dep)
                            syntax = list(zip(p, dependences_p))

                else:
                    doc = nlp(turn_["text"])
                    dependences = []
                    for token in doc:
                        dep = token.dep_
                        dependences.append(dep)
                        turn = turn_["text"].split(" ")
                        syntax = list(zip(turn, dependences))
            for lex_dep, lex_dep2 in zip(syntax, syntax[1:]):
                word, word_next = get_lexeme(lex_dep, lex_dep2)
                sintagma, sintagma_next = lex_dep[1], lex_dep2[1]
                word_parse, word_parse2 = get_morph_tags(lex_dep, lex_dep2)
                # complex rules
                is_nounroot = nounroot(word_parse, sintagma)
                is_verb_inf = verb_inf(word_parse, sintagma_next)
                is_verb_obj = verb_obj(word_parse, sintagma_next)
                is_npro_descr = npro_descr(word_parse, word_parse2)
                is_verb_constr = verb_constr(word_parse, word_parse2)
                if word_parse.tag.POS == "NPRO" and word_parse.tag.person == "1per":
                    npro.append(word)
                elif is_nounroot:
                    nounroot.append(word)
                elif is_verb_inf:
                    verb_plus_inf.append(word)
                    verb_plus_inf.append(word)
                elif is_verb_obj:
                    verb_plus_obj.append(word)
                    verb_plus_obj.append(word_next)
                elif sintagma == "xcomp" and sintagma_next == "obj":
                    inf_plus_obj.append(word)
                    inf_plus_obj.append(word_next)
                elif sintagma == "obl":
                    obl.append(word)
                elif word_parse.tag.POS == "INFN":
                    xcomp.append(word)
                elif sintagma == "csubj":
                    csubj.append(word)
                elif sintagma == "nmod":
                    nmod.append(word)
                elif word_parse.tag.POS == "NOUN" and word_parse.tag.case != "nomn":
                    noun_case.append(word)
                elif word_parse.tag.POS == "ADJF" and word_parse2.tag.POS == "NOUN":
                    adj_noun.append(word)
                    adj_noun.append(word_next)
                elif is_verb_constr:
                    verb_constr.append(word)
                elif word_parse.tag.POS == "ADJF":
                    adj.append(word)
                elif word_parse.tag.POS == "PRTF":
                    prtf.append(word)
                elif word_parse.tag.POS == "PRTF" and word_parse2.tag.POS == "NOUN":
                    prtf_noun.append(word)
                    prtf_noun.append(word_next)
                elif word_parse.tag.POS == "NUMR":
                    numr.append(word)
                    numr.append(word_next)
                elif word == "[0-9]*" and word_parse2.tag.POS == "NOUN":
                    digit_noun.append(word)
                    digit_noun.append(word_next)
                elif is_npro_descr:
                    npro_descripive.append(word_next)
                elif sintagma == "cc" and sintagma_next == "conj":
                    cc_conj.append(word)
                    cc_conj.append(word_next)
                elif sintagma == "pobj":
                    pobj.append(word)
    return (
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
    )
