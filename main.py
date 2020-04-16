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

import pandas as pd
import numpy as np
from numpy import *
import json
import gensim
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
from sklearn.feature_extraction.text import CountVectorizer


number_of_topics = 6
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

def get_vector_applicant():
    counter = 1
    with open('Features.txt', 'r') as file_handler:
        features_text = file_handler.readlines()
    # print(features_text)

    dict2 = {}
    for i in features_text:
        strWithoutComma = i.replace(","," ")
        strWithoutReg = strWithoutComma.lower()
        # value[value.index(word)] = re.sub(r"[^a-zA-Z]", " ", word)
        strWithoutSymbol = re.sub(r"[^a-zA-Z]", " ", strWithoutReg)
        tok_text = word_tokenize(strWithoutSymbol)
        pos_text = pos_tag(tok_text)
        sent_clean = [x for (x,y) in pos_text if (y not in ('PRP') and y not in ('DT') and y not in ('CC')) and y not in ('VB')]
        dict2.update({counter : sent_clean})
        counter += 1

    for key,value in dict2.items():
        for word in value:
            if (word =="I" or word =="i"):
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
    with open("FeaturesWithout_Reg_Comma_PRP.txt","r") as text:
        tempdict = json.load(text)
        finder = BigramCollocationFinder.from_documents(tempdict.values())
    print(finder.nbest(bigram_measures.raw_freq,15))
    #amountWords = finder.word_fd.items()
    sort_amountWords = sorted(finder.word_fd.items(), key=operator.itemgetter(1))
    print(sort_amountWords)

    synonyms = {}
    lemmas = []
    for word,number in sort_amountWords:
        if number > 15:
            lemmas.clear()
            for syn in wn.synsets(word):
                for l in syn.lemmas():
                    lemmas.append(l.name())
            synonyms.update({word : lemmas.copy()})
    print(synonyms)


    for tuple2 in sort_amountWords:
        for origWord,synList in synonyms.items():
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
    for word,number in sort_amountWords_dict.items():
        if number < 15:
            hyp.clear()
            for i,j in enumerate(wn.synsets(word)):
                if i < 2:
                    x = list(chain(*[l.lemma_names() for l in j.hyponyms()]))
                    hyp.append(x)
            Hyponyms.update({word : hyp.copy()})
    print(Hyponyms)

    for key,value in Hyponyms.items():
        for lst in value:
            for word in lst:
                for word_dict,number in sort_amountWords_dict.items():
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
    #print(dict_sort_amountWords)

    result_dict_popural_feature = {}
    alpha = 0.0085
    for key,value in dict_sort_amountWords.items():
        if (value/count_all_words) >= alpha:
            result_dict_popural_feature.update({key : value})
    print(result_dict_popural_feature)

    amount_items = 0
    for kye in result_dict_popural_feature.keys():
        amount_items+=1
    print(amount_items)

    countr = 0
    pos_text1 = pos_tag(result_dict_popural_feature.keys())
    sent_clean = [x for (x, y) in pos_text1 if
                  (y != ('PRP') and y != ('DT') and y != ('CC') and y != ('VB') and y !=('VBP') and y !=('VBN') and y !=('VBG') and y !=('RB'))]

    tempDict = dict(result_dict_popural_feature)
    for key1, value in tempDict.items():
        if key1 not in sent_clean:
            del result_dict_popural_feature[key1]
    #print(result_dict_popural_feature)


    tempDict_1 = dict(result_dict_popural_feature)
    for key,value in tempDict_1.items():
        countrr = 0
        for x in key:
            countrr+=1
        #print(countrr)
        if countrr < 3:
            del result_dict_popural_feature[x]
    print(result_dict_popural_feature)

    lst_features = list(result_dict_popural_feature.keys())
    print(lst_features)

    with open("Features_popular.txt", "w") as Features_popular:
        # for key,value in dict2.items():
        json.dump(lst_features, Features_popular)
        # featuresWithIdFile.write("{}: {}\n".format(key,value))

def get_full_vector():
    with open('trunc_Fashion.json', 'r') as f:
        jsonData = json.load(f)

    dict1 = {}
    for i in jsonData:
        if (i["overall"] >= 1) and (i["overall"] <= 5):
            tok_text = word_tokenize(i["reviewText"])
            customStopWords = set(stopwords.words('english') + list(punctuation))
            WordsStopResult = [word for word in tok_text if word not in customStopWords]
            lemmitazer_output = lemmatize_sentence(WordsStopResult)
            dict1.update({i["id"]:lemmitazer_output})

    with open('Initial_reviews_clean.txt', 'w') as Initial_reviews_clean:
        for key, val in dict1.items():
            Initial_reviews_clean.write('{}:{}\n'.format(key, val))  # Dict.txt - file with clear review's text from future recycle

    with open("Initial_reviews_clean.txt", "r") as Initial_reviews_clean:
        documents = Initial_reviews_clean.read().splitlines()
    # print(documents)

    count_vectorizer = CountVectorizer()
    bag_of_words = count_vectorizer.fit_transform(documents)
    feature_names = count_vectorizer.get_feature_names()
    pprint(pd.DataFrame(bag_of_words.toarray(), columns=feature_names), open("matrix.txt", "w"))
    # with open("matrix.txt","w") as f: #вывод проиндексированной матрицы в файл
    #     for i in range(len(bag_of_words.toarray())):
    #         for j in range(len(bag_of_words.toarray()[i])):
    #             f.write(str(bag_of_words.toarray()[i][j]))
    #         f.write("\n")

    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_vectorizer = TfidfVectorizer()
    values = tfidf_vectorizer.fit_transform(documents)
    feature_names = tfidf_vectorizer.get_feature_names()
    print(pd.DataFrame(values.toarray(), columns=feature_names))
    # with open("matrix.txt","w") as f:
    #     for i in range(len(values.toarray())):
    #         for j in range(len(values.toarray()[i])):
    #             f.write(str(values.toarray()[i][j]))
    #         f.write("\n")


get_word_applicant()
get_vector_applicant()
get_full_vector()