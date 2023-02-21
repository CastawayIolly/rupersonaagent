import json
import pymorphy2

from nltk.stem.snowball import SnowballStemmer
import spacy

nlp = spacy.load("ru_core_news_lg")
stemmer = SnowballStemmer("russian")
morph = pymorphy2.MorphAnalyzer()

# lexical constructions with stemming/ without stemming
# with stemming by default

alp = list('qwertyuiopasdfghjklzxcvbnm')
L = {}
# ngram=[]
with open('data/new.jsonl', 'w', encoding="utf-8") as out:
    with open('data/TolokaPersonaChat_gk_1_500_withoutgender.jsonl',
              'r', encoding="utf-8") as inp:
        for line in inp:
            line = json.loads(line)
            dialog = line['dialog']
            for turn_i, turn_ in enumerate(dialog):
                doc = nlp(turn_['text'])
                dependences = []
                ngram = []
                for token in doc:
                    dep = token.dep_
                    dependences.append(dep)
                    turn = turn_['text'].split(' ')
                    syntax = list(zip(turn, dependences))
                    for word2, nextWord2 in zip(syntax, syntax[1:]):
                        word = word2[0]
                        sintagma = word2[1]
                        word = [letter for letter in word
                                if letter.isalpha() and letter not in alp]
                        word = ''.join(word)
                        word_parse = morph.parse(word)[0]
                        word_next = nextWord2[0]
                        sintagma_next = nextWord2[1]
                        word_next = [letter for letter in word_next
                                     if letter.isalpha() and letter not in alp]
                        word_next = ''.join(word_next)
                        word_parse2 = morph.parse(word_next)[0]
                        if word_parse.tag.POS == "NOUN":
                            ngram.append(word_parse.normal_form)
                        elif word_parse.tag.POS == "INFN":
                            ngram.append(word_parse.normal_form)
                        elif word_parse.tag.POS == "VERB" \
                                and word_parse.tag.person == "1per" \
                                and (word_parse2.tag.POS != "PREP"
                                     or word_parse2.tag.POS != "CONJ"
                                     or word_parse2.tag.POS == "ADVB"):
                            ngram.append(word_parse2.normal_form)
                        elif word_parse.tag.POS == "VERB" and \
                                word_parse.tag.person == "1per" and \
                                (sintagma_next == "obj"
                                 or sintagma_next == "xcomp"):
                            ngram.append(word_parse2.normal_form)
                        elif sintagma == "xcomp" \
                                and sintagma_next == "obj":
                            ngram.append(word_parse.normal_form)
                            ngram.append(word_parse2.normal_form)
                        elif sintagma == "obl" or sintagma == "nmod":
                            ngram.append(word_parse.normal_form)
                        elif word_parse.tag.POS == "ADJF":
                            ngram.append(word_parse.normal_form)
                        elif word_parse.tag.POS == "PRTF":
                            ngram.append(word_parse.normal_form)
                        elif (word_parse.tag.POS == "ADJF" or
                              word_parse.tag.POS == "PRTF") \
                                and word_parse2.tag.POS == "NOUN":
                            ngram.append(word_parse.normal_form)
                            ngram.append(word_parse2.normal_form)
                        elif word_parse.tag.POS == "NUMR":
                            ngram.append(word_parse.normal_form)
                            ngram.append(word_parse2.normal_form)
                        elif word == "[0-9]*":
                            ngram.append(word_parse.normal_form)
                            ngram.append(word_parse2.normal_form)
                        elif (word_parse.tag.POS == "NPRO" and
                              word_parse.tag.person == "1per") and \
                                word_parse2.tag.POS == "NOUN":
                            ngram.append(word_parse2.normal_form)

                    turn = ngram
                    # WITH STEMS
                    # ngram=[stemmer.stem(word) for word in ngram]
                    persona_ = line['persons'][turn_['person']]
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
                            for word2, nextWord2 in zip(syntax, syntax[1:]):
                                word = word2[0]
                                sintagma = word2[1]
                                word = [letter for letter in word
                                        if letter.isalpha()
                                        and letter not in alp]
                                word = ''.join(word)
                                word_parse = morph.parse(word)[0]
                                word_next = nextWord2[0]
                                sintagma_next = nextWord2[1]
                                word_next = [letter for letter
                                             in word_next
                                             if letter.isalpha()
                                             and letter not in alp]
                                word_next = ''.join(word_next)
                                word_parse2 = morph.parse(word_next)[0]
                                if (word_parse.tag.POS == "NOUN"
                                        and (sintagma == "ROOT"
                                             or sintagma == "nsubj")):
                                    ngrams.append(word_parse.normal_form)
                                elif word_parse.tag.POS == "INFN":
                                    ngrams.append(word_parse.normal_form)
                                elif word_parse.tag.POS == "VERB" and \
                                        word_parse.tag.person == "1per"\
                                        and (sintagma_next == "obj"
                                             or sintagma_next == "xcomp"):
                                    ngrams.append(word_parse.normal_form)
                                    ngrams.append(word_parse2.normal_form)
                                elif sintagma == "xcomp" \
                                        and sintagma_next == "obj":
                                    ngrams.append(word_parse.normal_form)
                                    ngrams.append(word_parse2.normal_form)
                                elif word_parse.tag.POS == "NOUN" \
                                        and word_parse.tag.case != "nomn":
                                    ngrams.append(word_parse.normal_form)
                                elif sintagma == "obl" or sintagma == "nmod":
                                    ngrams.append(word_parse.normal_form)
                                elif (word_parse.tag.POS == "NPRO"
                                      and word_parse.tag.person == "1per")\
                                        and word_parse2.tag.POS == "NOUN":

                                    ngrams.append(word_parse2.normal_form)

                                elif word_parse.tag.POS == "PRTF":
                                    ngrams.append(word_parse.normal_form)
                                elif word_parse.tag.POS == "NUMR":
                                    ngrams.append(word_parse.normal_form)
                                    ngrams.append(word_parse2.normal_form)
                                elif word == "[0-9]*":
                                    ngrams.append(word_parse.normal_form)
                                    ngrams.append(word_parse2.normal_form)

                                elif word_parse.tag.POS == "ADJF":
                                    ngrams.append(word_parse.normal_form)
                                elif (word_parse.tag.POS == "ADJF" or
                                      word_parse.tag.POS == "PRTF") \
                                        and word_parse2.tag.POS == "NOUN":
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

                    if len(L)-1 < len(golden_p):
                        L[len(golden_p)] = 1
                    else:
                        L[len(golden_p)] += 1
                    dialog[turn_i]['gk'] = list(golden_p)

            line = json.dumps(line, ensure_ascii=False)
            out.write(line+'\n')
