'''
POS tag list
CC coordinating conjunction
CD cardinal digit
DT determiner
EX existential there (like: “there is” … think of it like “there exists”)
FW foreign word
IN preposition/subordinating conjunction
JJ adjective ‘big’
JJR adjective, comparative ‘bigger’
JJS adjective, superlative ‘biggest’
LS list marker 1)
MD modal could, will
NN noun, singular ‘desk’
NNS noun plural ‘desks’
NNP proper noun, singular ‘Harrison’
NNPS proper noun, plural ‘Americans’
PDT predeterminer ‘all the kids’
POS possessive ending parent’s
PRP personal pronoun I, he, she
PRP$ possessive pronoun my, his, hers
RB adverb very, silently,
RBR adverb, comparative better
RBS adverb, superlative best
RP particle give up
TO, to go ‘to’ the store.
UH interjection, errrrrrrrm
VB verb, base form take
VBD verb, past tense took
VBG verb, gerund/present participle taking
VBN verb, past participle taken
VBP verb, sing. present, non-3d take
VBZ verb, 3rd person sing. present takes
WDT wh-determiner which
WP wh-pronoun who, what
WP$ possessive wh-pronoun whose
WRB wh-abverb where, when
'''

import nltk.sentiment.sentiment_analyzer
import pandas as pd
import numpy as np
from numpy import *
import json
import gensim
import string
import random
import requests
import bs4 as bs
from bs4 import BeautifulSoup
import urllib.request
from gensim import corpora
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords
from string import punctuation
from nltk.collocations import *
from nltk.corpus import wordnet as wn
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import brown
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.chunk import api
from nltk.chunk.api import ChunkParserI
from nltk import RegexpParser
from nltk.tag import pos_tag
from nltk import pos_tag_sents
import scipy
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from nltk.tokenize import PunktSentenceTokenizer
from smart_open import smart_open
import os
import os.path
from pprint import pprint
from nltk.corpus import twitter_samples
from itertools import chain
import re
import operator
import sys
from collections import Counter
import sklearn
from sklearn.model_selection import train_test_split as train
from sklearn.feature_extraction.text import CountVectorizer
import heapq
# from PyDictionary import PyDictionary
# from py_thesaurus import Thesaurus
import gensim, logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import csv

words = 20
stemmer = SnowballStemmer('english')


def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence


def get_word_applicant():
    with open('trunc_Fashion.json', 'r') as f:
        jsonData = json.load(f)

    file_handler = open("Features.txt", "w")
    dict1 = {}
    for i in jsonData:
        # if (i["overall"] > 1) and (i["overall"] < 5):
        if (i["overall"] >= 1) and (i["overall"] <= 5):
            tok_text = word_tokenize(i["reviewText"])
            customStopWords = set(stopwords.words('english') + list(punctuation))
            WordsStopResult = [word for word in tok_text if word not in customStopWords]
            lemmitazer_output = lemmatize_sentence(tok_text)
            pos_text = pos_tag(lemmitazer_output)
            ChunkGramma = (r"Chunk:{(<NN>|<NNS>)+(<VB>|<VBD>|<MD>|<VBP>)?<DT>?(<NN>|<NNS>)} " "\n"
                           r"{(<JJ>|<JJR>|<JJS>|<,>)+(<NN>|<NNS>)}" "\n"
                           r"{<EX>(<JJ>|<JJR>|<JJS>)?((<NN>|<NNS>))?}" "\n"
                           r"{<PRP>(<VB>|<VBD>|<MD>|<VBP>)<RB>?(<JJ>|<JJR>|<JJS>)+<CC>?(<JJ>|<JJR>|<JJS>|<NN>)?}" "\n"
                           r" {<DT>(<VB>|<VBD>|<MD>|<VBP>)<DT>?(<RBS>|<RB>)?(<JJ>|<JJR>|<JJS>|<NN>|<,>)+}" "\n"
                           r"{(<JJ>|<JJR>|<JJS>)+(<NN>|<NNS>)}" "\n"
                           r"{(<JJ>|<JJR>|<JJS>|<,>)+(<NN>|<NNS>|<,>)+<CC>?<DT>?(<JJ>|<JJR>|<JJS>|<,>)+(<NN>|<NNS>|<,>)+}" "\n"
                           r"{<PRP>(<VB>|<VBD>|<MD>|<VBP>)<RB>?<DT>?(<JJ>|<JJR>|<JJS>|<NN>|<,>)+}" "\n"
                           r"{(<JJ>|<JJR>|<JJS>|<,>)+<CC>?(<JJ>|<JJR>|<JJS>|<,>)+}" "\n"
                           r"{<PRP>(<VB>|<VBD>|<MD>|<VBP>)(<RB>|<RBR>|<RBS>|<,>)+}" "\n"
                           r"{(<NN>|<NNS>|<,>)+<CC>?(<NN>|<NNS>|<,>)(<VB>|<VBD>|<MD>|<VBP>)?(<JJ>|<JJR>|<JJS>)+<CC>?(<JJ>|<JJR>|<JJS>|<NN>)?}" "\n"
                           r"{(<JJ>|<JJR>|<JJS>|<RB>|<,>)+<IN><DT>(<NN>|<NNS>)<CC>?(<NN>|<NNS>)}" "\n"
                           r"")
            chunkParser = nltk.RegexpParser(ChunkGramma)
            chunked = chunkParser.parse(pos_text)
            # chunked.draw()

            for subtree in chunked.subtrees():
                if subtree.label() == 'Chunk':
                    t = subtree
                    t = ' '.join(word for word, pos in t.leaves())
                    file_handler.write(str(t) + "\n")
            dict1.update({i["id"]: pos_text})

    file_handler.close()
    return t


def dict_stop_word():
    myself_dict_stop = ['good', 'great', 'cool', 'ok', 'love', 'hate', 'i', 'perfect', 'kind', 'well', 'nice',
                        'one', 'help', 'have', 'some', 'want', 'put', 'home', 'even', 'went', 'try', 'take']
    return myself_dict_stop


def get_vector_applicant():
    counter = 1
    with open('Features.txt', 'r') as file_handler:
        features_text = file_handler.readlines()
    # print(features_text)

    dict2 = {}
    for i in features_text:
        strWithoutComma = i.replace(",", " ")
        strWithoutReg = strWithoutComma.lower()
        # value[value.index(word)] = re.sub(r"[^a-zA-Z]", " ", word)
        strWithoutSymbol = re.sub(r"[^a-zA-Z]", " ", strWithoutReg)
        tok_text = word_tokenize(strWithoutSymbol)
        pos_text = pos_tag(tok_text)
        sent_clean = [x for (x, y) in pos_text if
                      (y not in ('PRP') and y not in ('DT') and y not in ('CC')) and y not in ('VB')]
        dict2.update({counter: sent_clean})
        counter += 1

    for key, value in dict2.items():
        for word in value:
            if (word == "I" or word == "i"):
                value.remove(word)

    dct = {}
    count_all_words = 0
    for key, value in dict2.items():
        for word in value:
            count_all_words += 1
    print(count_all_words)

    with open("FeaturesWithout_Reg_Comma_PRP.txt", "w") as featuresWithout_Reg_Comma_PR:
        # for key,value in dict2.items():
        json.dump(dict2, featuresWithout_Reg_Comma_PR)
        # featuresWithIdFile.write("{}: {}\n".format(key,value))

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    with open("FeaturesWithout_Reg_Comma_PRP.txt", "r") as text:
        tempdict = json.load(text)
        finder = BigramCollocationFinder.from_documents(tempdict.values())
    print(finder.nbest(bigram_measures.raw_freq, 15))
    # amountWords = finder.word_fd.items()
    sort_amountWords = sorted(finder.word_fd.items(), key=operator.itemgetter(1))
    print(sort_amountWords)

    synonyms = {}
    lemmas = []
    for word, number in sort_amountWords:
        if number > 15:
            lemmas.clear()
            for syn in wn.synsets(word):
                for l in syn.lemmas():
                    lemmas.append(l.name())
            synonyms.update({word: lemmas.copy()})
    print(synonyms)

    for tuple2 in sort_amountWords:
        for origWord, synList in synonyms.items():
            if tuple2[0] in synList:
                lst = list(tuple2)
                lst[0] = origWord
                sort_amountWords[sort_amountWords.index(tuple2)] = tuple(lst)
                break
    print(sort_amountWords)

    sort_amountWords_dict = dict(sort_amountWords)
    c = Counter()
    for word in sort_amountWords_dict:
        c.update(word)
    print("result:")
    print(sort_amountWords_dict)

    Hypernyms = {}
    Hyponyms = {}
    hyp = []
    for word, number in sort_amountWords_dict.items():
        if number < 15:
            hyp.clear()
            for i, j in enumerate(wn.synsets(word)):
                if i < 2:
                    x = list(chain(*[l.lemma_names() for l in j.hyponyms()]))
                    hyp.append(x)
            Hyponyms.update({word: hyp.copy()})
    print(Hyponyms)

    for key, value in Hyponyms.items():
        for lst in value:
            for word in lst:
                for word_dict, number in sort_amountWords_dict.items():
                    if number > 15:
                        if word_dict in lst:
                            sort_amountWords_dict.update({key: number})
                            break
    sorted_x = sorted(sort_amountWords_dict.items(), key=operator.itemgetter(1))
    print(sorted_x)

    with open("Features_on_Start.txt", "w") as Features_on_Start:
        # for key,value in dict2.items():
        json.dump(sorted_x, Features_on_Start)
        # featuresWithIdFile.write("{}: {}\n".format(key,value))

    dict_sort_amountWords = dict(sorted_x)
    # print(dict_sort_amountWords)

    result_dict_popural_feature = {}
    alpha = 0.0085
    for key, value in dict_sort_amountWords.items():
        if (value / count_all_words) >= alpha:
            result_dict_popural_feature.update({key: value})
    print(result_dict_popural_feature)

    amount_items = 0
    for kye in result_dict_popural_feature.keys():
        amount_items += 1
    print(amount_items)

    countr = 0
    pos_text1 = pos_tag(result_dict_popural_feature.keys())
    sent_clean = [x for (x, y) in pos_text1 if
                  (y != ('PRP') and y != ('DT') and y != ('CC') and y != ('VB') and y != ('VBP') and y != (
                      'VBN') and y != ('VBG') and y != ('RB'))]

    tempDict = dict(result_dict_popural_feature)
    for key1, value in tempDict.items():
        if key1 not in sent_clean:
            del result_dict_popural_feature[key1]
    # print(result_dict_popural_feature)

    tempDict_1 = dict(result_dict_popural_feature)
    for key, value in tempDict_1.items():
        countrr = 0
        for x in key:
            countrr += 1
        # print(countrr)
        if countrr < 3:
            del result_dict_popural_feature[x]
    # print(result_dict_popural_feature)

    stop_word_my = dict_stop_word()
    tempDict2 = dict(result_dict_popural_feature)
    for key2 in tempDict2.keys():
        if key2 in stop_word_my:
            del result_dict_popural_feature[key2]

    lst_features = list(result_dict_popural_feature.keys())
    print(lst_features)

    with open("Features_popular.txt", "w") as Features_popular:
        # for key,value in dict2.items():
        json.dump(lst_features, Features_popular)
        # featuresWithIdFile.write("{}: {}\n".format(key,value))


def syn():
    with open('Features_popular.txt', 'r') as Features_popular:
        features_text = json.load(Features_popular)

    synonyms = {}
    lemmas = []
    for word in features_text:
        lemmas.clear()
        for syn in wn.synsets(word):
            for l in syn.lemmas():
                lemmas.append(l.name())
        synonyms.update({word: lemmas.copy()})
    return synonyms


def hypo():
    with open('Features_popular.txt', 'r') as Features_popular:
        features_text = json.load(Features_popular)

    Hyponyms = {}
    hyp = []
    for word in features_text:
        hyp.clear()
        for i, j in enumerate(wn.synsets(word)):
            if i < 1:
                x = list(chain(*[l.lemma_names() for l in j.hyponyms()]))
                hyp.append(x)
        Hyponyms.update({word: hyp.copy()})
    return Hyponyms


def hype():
    with open('Features_popular.txt', 'r') as Features_popular:
        features_text = json.load(Features_popular)

    Hypernyms = {}
    hype = []
    for word in features_text:
        hype.clear()
        for i, j in enumerate(wn.synsets(word)):
            if i < 2:
                x = list(chain(*[l.lemma_names() for l in j.hypernyms()]))
                hype.append(x)
        Hypernyms.update({word: hype.copy()})
    return Hypernyms


def semantic_score(word1, word2):
    try:
        w1 = wn.synset("%s.n.01" % (word1))
        w2 = wn.synset("%s.n.01" % (word2))
        return wn.wup_similarity(w1, w2, simulate_root=False)
    except:
        return 0


def NegativeWord():
    with open('trunc_Fashion.json', 'r') as f:
        jsonData = json.load(f)

    tag_negative_words = list(copy(jsonData))
    for i in tag_negative_words:
        text = i["reviewText"].split()
        analysis = nltk.sentiment.util.mark_negation(text)
        customStopWords = set(stopwords.words('english') + list(punctuation))
        WordsStopResult = [word for word in analysis if word not in customStopWords]
        lemmitazer_output = lemmatize_sentence(WordsStopResult)
        i.update({"reviewText": lemmitazer_output})
        del (i["label"])

    # print(tag_negative_words)

    with open("Tag_nagative.txt", "w") as tag_negative:
        json.dump(tag_negative_words, tag_negative, indent=4)

    return tag_negative_words


def get_full_vector():
    with open('trunc_Fashion.json', 'r') as f:
        jsonData = json.load(f)

    corpus = []
    for i in jsonData:
        corpus.append(i["reviewText"])
    print(corpus)

    # corpus_new = map(lambda x: x.lower(), corpus)
    # print(corpus_new)
    for i in range(len(corpus)):
        corpus[i] = corpus[i].lower()
        corpus[i] = re.sub(r'\W', ' ', corpus[i])
        corpus[i] = re.sub(r'\s+', ' ', corpus[i])
    print(corpus)
    # print(len(corpus))

    with open('Features_popular.txt', 'r') as Features_popular:
        features_text = json.load(Features_popular)
    print(features_text)

    synonym = syn()
    sentence_vectors = []
    tag_neg = NegativeWord()
    for i in tag_neg:
        sentence_tokens = i["reviewText"]
        # sentence_tokens = nltk.word_tokenize(sentence)
        sent_vec = {}
        sent_vec.update({"overall": i["overall"]})
        for token in features_text:
            flag = False
            if token in sentence_tokens:
                if token + '_NEG' in sentence_tokens:
                    sent_vec.update({token: -1})
                    flag = True
                else:
                    sent_vec.update({token: 1})
                    flag = True
            else:
                for syno in synonym[token]:
                    if syno in sentence_tokens:
                        count = semantic_score(syno, sentence_tokens[sentence_tokens.index(syno)])
                        if (count >= 0.5):
                            sent_vec.update({token: 1})
                            flag = True
                            break
                    else:
                        if syno + "_NEG" in sentence_tokens:
                            count = semantic_score(syno, sentence_tokens[sentence_tokens.index(syno + "_NEG")][:-4])
                            if (count >= 0.5):
                                sent_vec.update({token: -1})
                                flag = True
                                break
            if not flag:
                sent_vec.update({token: 0})
        sentence_vectors.append(sent_vec)
    # print(sentence_vectors)
    # sentence_vectors = np.asarray(sentence_vectors)

    print(sentence_vectors)

    with open("Data_vector_reviews.csv", "w", newline="") as file:
        columns = sentence_vectors[0].keys()
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        # запись нескольких строк
        writer.writerows(sentence_vectors)

    # sentence_vectors = np.matrix(sentence_vectors)

    # with open('Matrix_vectors.txt', 'wb') as f:
    #     for line in sentence_vectors:
    #         np.savetxt(f, line, fmt='%.2f')
    # #print(sentence_vectors)

    return sentence_vectors


# from gensim.models import Word2Vec
# from nltk.corpus import gutenberg
# def word2_vec():
#     sentences = [['first', 'sentence'], ['second', 'sentence']]
#     # train word2vec on the two sentences
#     model = gensim.models.Word2Vec(sentences, min_count=1)
#     sim = model.wv.most_similar('first')
#     for w,s in sim:
#         print(w,s)
# def Thesaurus():
# input_word = "work"
# t = Thesaurus(input_word)
# print(t.get_synonym(pos='adj'))
# for i, j in enumerate(wn.synsets('dog')):
#     print("Hypernyms:", ", ".join(list(chain(*[l.lemma_names() for l in j.hypernyms()]))))
#     print("Hyponyms:", ", ".join(list(chain(*[l.lemma_names() for l in j.hyponyms()]))))


from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import math


def Logistic_Reression():
    max_epoch = 20
    data = pd.read_csv('Data_vector_reviews.csv')
    X = data.values[::, 1:21]
    y = data.values[::, 0:1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(len(X_train), " train +", len(X_test), "test")

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_index, test_index in split.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    print(data["overall"].value_counts() / len(data))

    # -----------------------------------------------Log_Regression--------------------------------------------------------
    lg_clf = LogisticRegression(penalty='l1', solver='liblinear')
    y_train = y_train.ravel()
    lg_clf.fit(X_train, y_train)

    lg_clf_prediction = lg_clf.predict(X_test)
    print("Прогнозы:", lg_clf_prediction)
    print("Метки:", list(y_test))
    print(accuracy_score(lg_clf_prediction, y_test))
    print(metrics.classification_report(y_test, lg_clf_prediction))
    print(metrics.confusion_matrix(y_test, lg_clf_prediction))


from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, log_loss


def Decision_Tree():
    data = pd.read_csv('Data_vector_reviews.csv')
    X = data.values[::, 1:21]
    y = data.values[::, 0:1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(len(X_train), " train +", len(X_test), "test")

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_index, test_index in split.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    print(data["overall"].value_counts() / len(data))

    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(X_train, y_train)
    y_pred = tree_clf.predict(X_test)
    # loss = log_loss(y_test, y_pred)
    # print(loss)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
def Forest_random():
    data = pd.read_csv('Data_vector_reviews.csv')
    X = data.values[::, 1:21]
    y = data.values[::, 0:1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(len(X_train), " train +", len(X_test), "test")

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    for train_index, test_index in split.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    print(data["overall"].value_counts() / len(data))

    model = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt')
    model.fit(X_train, y_train)

    model_predict = model.predict(X_test)
    print(confusion_matrix(y_test, model_predict))
    print(classification_report(y_test, model_predict))


# get_word_applicant()
# get_vector_applicant()
# get_full_vector()
# Logistic_Reression()
#Decision_Tree()
Forest_random()
