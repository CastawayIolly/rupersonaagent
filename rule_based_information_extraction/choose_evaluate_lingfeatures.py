
import pandas as pd
import json
import re
import tqdm
import pymorphy2

from pymorphy2.units.base import BaseAnalyzerUnit
from nltk.stem.snowball import SnowballStemmer 
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import spacy
from spacy.lang.ru.examples import sentences 

nlp = spacy.load("ru_core_news_lg")
stemmer = SnowballStemmer("russian") 
morph = pymorphy2.MorphAnalyzer()

#get reference text
result_human=[]
result_human_text=[]
with open(r'data/TolokaPersonaChat_gk_1_500_withoutgender.jsonl', 'r', encoding="utf-8") as inp:
      for line in inp:
            line = json.loads(line)
            dialog = line['dialog']
            for turn_i, turn_ in enumerate(dialog):
                  turn = turn_['gk']
                  turn1=turn_['text']
                  if len(turn) != 0: 
                        turn="yes"
                  else:
                        turn="no"
                  result_human.append(turn)
                  result_human_text.append(turn1)


# features
#dialog replicas
npro, nounroot, obl, xcomp, csubj, nmod, noun_case, adj_noun, adj, prtf, pobj=[],[], [], [],[], [],[], [], [], [], [], [], [], []
verb_plus_inf, verb_constr, npro_descripive=[], [], []
verb_plus_obj, inf_plus_obj, prtf_noun, digit_noun, numr, cc_conj=[], [], [], [], [], []

#persona description
npro_p, nounroot_p, obl_p, xcomp_p, csubj_p, nmod_p, noun_case_p, adj_noun_p, adj_p, prtf_p, pobj_p=[],[], [], [],[], [],[], [], [], [], []
verb_plus_inf_p, verb_constr_p, npro_descripive_p=[], [], []
verb_plus_obj_p, inf_plus_obj_p, prtf_noun_p, digit_noun_p, numr_p, cc_conj_p=[], [], [], [], [], []




alp=list('qwertyuiopasdfghjklzxcvbnm')
#get results from dialog replicas
L={}

with open('data/TolokaPersonaChat_gk_1_500_withoutgender.jsonl', 'r', encoding="utf-8") as inp:
    for line in inp:
        line = json.loads(line)
        dialog = line['dialog']
        for turn_i, turn_ in enumerate(dialog):
            doc = nlp(turn_['text'])
            dependences=[]
            for token in doc:
                dep=token.dep_
                dependences.append(dep)
                turn = turn_['text'].split(' ')
                
                syntax=list(zip(turn, dependences))
                for i in range(len(syntax)):
                    syntax1=syntax[i]
                    try:
                        syntax2=syntax[i+1]
                    except:
                        syntax2="null"
                    word=syntax1[0]
                    sintagma=syntax1[1]
                    word = [l for l in word if l.isalpha() and l not in alp]
                    word = ''.join(word)
                    word_parse= morph.parse(word)[0] 
                    try: 
                        word_next=syntax2[0]
                        sintagma_next=syntax2[1]
                        word_next = [l for l in word_next if l.isalpha() and l not in alp]
                        word_next = ''.join(word_next)
                        word_parse2= morph.parse(word_next)[0] 
                    except IndexError:
                        word_next = 'null'
                        sintagma_next='null'

                    if (word_parse.tag.POS=="NPRO"  and word_parse.tag.person=="1per" ): #or (word_parse.tag.POS=="VERB" and word_parse.tag.mood=="indc")) and word_parse.tag.person=="1per" :
                        npro.append(word)
                    elif (word_parse.tag.POS=="NOUN" and sintagma=="ROOT") or (word_parse.tag.POS=="NOUN" and word_parse.tag.animacy=="anim" and word_parse.tag.case=="nomn") :
                        nounroot.append(word)
                    elif  (word_parse.tag.POS=="VERB" and word_parse.tag.person=="1per" and word_parse.tag.number=="sing") and sintagma_next=="xcomp" :
                        verb_plus_inf.append(word)
                        verb_plus_inf.append(word)
                    elif (word_parse.tag.POS=="VERB" and word_parse.tag.person=="1per" and word_parse.tag.number=="sing") and sintagma_next=="obj":
                        verb_plus_obj.append(word)
                        verb_plus_obj.append(word_next)
                    elif  sintagma=="xcomp" and sintagma_next=="obj":
                        inf_plus_obj.append(word)
                        inf_plus_obj.append(word_next)
                    elif sintagma=="obl" :
                        obl.append(word)
                    elif word_parse.tag.POS=="INFN" :
                        xcomp.append(word)
                    elif sintagma=="csubj" :
                        csubj.append(word)
                    elif sintagma=="nmod":
                        nmod.append(word)
                    elif word_parse.tag.POS=="NOUN" and word_parse.tag.case!="nomn" :
                        noun_case.append(word)
                    elif word_parse.tag.POS=="ADJF" and word_parse2.tag.POS=="NOUN" :
                        adj_noun.append(word)
                        adj_noun.append(word_next)
                    elif (word_parse.tag.POS=="VERB" and word_parse.tag.person=="1per" and word_parse.tag.number=="sing")and (word_parse2.tag.POS!="PREP" or word_parse2.tag.POS!="CONJ" or word_parse2.tag.POS=="ADVB"):
                        verb_constr.append(word)
                    elif word_parse.tag.POS=="ADJF" :
                        adj.append(word)
                    elif word_parse.tag.POS=="PRTF" :
                        prtf.append(word)
                    elif word_parse.tag.POS=="PRTF" and word_parse2.tag.POS=="NOUN" :
                        prtf_noun.append(word)
                        prtf_noun.append(word_next)
                    elif word_parse.tag.POS=="NUMR" :
                        numr.append(word)
                        numr.append(word_next)
                    elif word=="[0-9]*" and word_parse2.tag.POS=="NOUN":
                        digit_noun.append(word)
                        digit_noun.append(word_next)
                    elif (word_parse.tag.POS=="NPRO" and word_parse.tag.person=="1per" and word_parse.tag.number=="sing") and (word_parse2.tag.POS=="NOUN" or word_parse2.tag.POS=="ADJF" or word_parse2.tag.POS=="PRTF" or word_parse2.tag.POS=="PRTS") :
                        npro_descripive.append(word_next)
                    elif sintagma=="cc" and sintagma_next=="conj": 
                        cc_conj.append(word)
                        cc_conj.append(word_next)
                    elif sintagma=="pobj": 
                        pobj.append(word)


#get results from persona decription

with open('data/TolokaPersonaChat_gk_1_500_withoutgender.jsonl', 'r', encoding="utf-8") as inp:
        for line in inp:
            line = json.loads(line)
            dialog = line['dialog']
            for turn_i, turn_ in enumerate(dialog):
                doc = nlp(turn_['text'])
                dependences=[]
                ngram=[]
                persona_ = line['persons'][turn_['person']]
                persona=[]
                for p in persona_:
                        doc_p = nlp(p)
                        dependences_p=[]
                        ngrams=[]
                        p = p.split()
                        for token in doc_p:
                            dep=token.dep_
                            dependences_p.append(dep)
                            syntax=list(zip(p, dependences_p))
                            
                            for word2, nextWord2 in zip(syntax, syntax[1:]):
                                word=word2[0]
                                sintagma=word2[1]
                                word = [l for l in word if l.isalpha() and l not in alp]
                                word = ''.join(word)
                                word_parse= morph.parse(word)[0]  
                                word_next=nextWord2[0]
                                sintagma_next=nextWord2[1]
                                word_next = [l for l in word_next if l.isalpha() and l not in alp]
                                word_next = ''.join(word_next)
                                word_parse2= morph.parse(word_next)[0]    

                                if (word_parse.tag.POS=="NOUN" and sintagma=="ROOT") or (word_parse.tag.POS=="NOUN" and word_parse.tag.animacy=="anim" and word_parse.tag.case=="nomn") :
                                    nounroot_p.append(word_parse.normal_form)
                                elif  (word_parse.tag.POS=="VERB" and word_parse.tag.person=="1per" and word_parse.tag.number=="sing") and sintagma_next=="xcomp" :
                                    verb_plus_inf_p.append(word_parse.normal_form)
                                    verb_plus_inf_p.append(word_parse2.normal_form)

                                elif (word_parse.tag.POS=="VERB" and word_parse.tag.person=="1per" and word_parse.tag.number=="sing") and sintagma_next=="obj":
                                    verb_plus_obj_p.append(word_parse.normal_form)
                                    verb_plus_obj_p.append(word_parse2.normal_form)
                                elif  sintagma=="xcomp" and sintagma_next=="obj":
                                    inf_plus_obj_p.append(word_parse.normal_form)
                                    inf_plus_obj_p.append(word_parse2.normal_form)
                                elif sintagma=="obl" :
                                    obl_p.append(word_parse.normal_form)
                                elif word_parse.tag.POS=="INFN" :
                                    xcomp_p.append(word_parse.normal_form)

                                elif sintagma=="csubj" :
                                    csubj_p.append(word_parse.normal_form)
                                elif sintagma=="nmod":
                                    nmod_p.append(word_parse.normal_form)
                                elif word_parse.tag.POS=="NOUN" and word_parse.tag.case!="nomn" :
                                    noun_case.append(word_parse.normal_form)
                                elif word_parse.tag.POS=="ADJF" and word_parse2.tag.POS=="NOUN" :
                                   adj_noun_p.append(word_parse.normal_form)
                                   adj_noun_p.append(word_parse2.normal_form)
                                elif (word_parse.tag.POS=="VERB" and word_parse.tag.person=="1per" and word_parse.tag.number=="sing")and (word_parse2.tag.POS!="PREP" or word_parse2.tag.POS!="CONJ" or word_parse2.tag.POS=="ADVB"):

                                    verb_constr_p.append(word_parse.normal_form)
                                elif word_parse.tag.POS=="ADJF" :
                                    adj_p.append(word_parse.normal_form)
                                elif word_parse.tag.POS=="PRTF" :
                                    prtf_p.append(word_parse.normal_form)
                                elif word_parse.tag.POS=="PRTF" and word_parse2.tag.POS=="NOUN" :
                                    prtf_noun_p.append(word_parse.normal_form)
                                    prtf_noun_p.append(word_parse2.normal_form)
                                elif word_parse.tag.POS=="NUMR" :
                                    numr_p.append(word_parse.normal_form)
                                    numr_p.append(word_parse2.normal_form)
                                elif word=="[0-9]*" and word_parse2.tag.POS=="NOUN":
                                    digit_noun_p.append(word_parse.normal_form)
                                    digit_noun_p.append(word_parse2.normal_form)
                                elif (word_parse.tag.POS=="NPRO" and word_parse.tag.person=="1per" and word_parse.tag.number=="sing") and (word_parse2.tag.POS=="NOUN" or word_parse2.tag.POS=="ADJF" or word_parse2.tag.POS=="PRTF" or word_parse2.tag.POS=="PRTS") :
                                    npro_descripive_p.append(word_parse2.normal_form)

                                elif sintagma=="cc" and sintagma_next=="conj": 
                                    cc_conj_p.append(word_parse.normal_form)
                                    cc_conj_p.append(word_parse2.normal_form)
                                elif sintagma=="pobj": 
                                    pobj_p.append(word_parse.normal_form)


#assign features with each sentence in dataframe
def assign_lexical_items(full_sentence, item_dialog, item_personadescription):
    #get items that are in person description and in dialogs
    item=[i for i, j in zip(item_dialog, item_personadescription) if i == j]
    assigned_item=[]
    item = list(set(item))

    for x in full_sentence:
        k=[x for w in item if w in x]
        assigned_item.append(k)

    item_tag=[]
    for i in range(len(assigned_item)):
        if len(assigned_item[i])==0:
            item_tag.append("no")
        else:
            item_tag.append("yes")
    return item_tag

#cc_conj_tag=assign_lexical_items(result_human_text, cc_conj, cc_conj_p) 
#etc


#count metrics for each feature
def count_metrics(actual, pred):
    f1=f1_score(actual, pred ,average='weighted')
    ac=accuracy_score(actual, pred)
    return [f1, ac]

#metrics_cc_conj2=count_metrics(result_human, cc_conj_tag)
#etc

#combine features with the highest f1, acuracy scores
def combine_feautures(feature1, feauture2):
    example=[]
    for i in feature1:
        for j in feauture2:
            #print(i)
            if (i=="no" and j=="yes") :
                x="yes"
            elif (i=="yes" and j=="no") :
                x="yes"
            elif (i=="yes" and j=="yes"):
                x="yes"
            elif (i=="no" and j=="no"):
                x="no"
        example.append(x)


