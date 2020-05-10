import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.collocations import *
from collections import Counter
from itertools import chain
import re
from nltk.corpus import wordnet as wn
import operator

#Function keeps the dict, wich consists of the most popular useless words.
def dict_stop_word():
    myself_dict_stop = ['good', 'great', 'cool', 'ok', 'love', 'hate', 'i', 'perfect', 'kind', 'well', 'nice',
                        'one', 'help', 'have', 'some', 'want', 'put', 'home', 'even', 'went', 'try', 'take']
    return myself_dict_stop

#Function choosess special features from applicants.
def get_vector_applicant():
    counter = 1
    with open('Features.txt', 'r', encoding = "utf_8_sig") as file_handler:
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

    with open("FeaturesWithout_Reg_Comma_PRP.txt", "w", encoding = "utf_8_sig") as featuresWithout_Reg_Comma_PR:
        # for key,value in dict2.items():
        json.dump(dict2, featuresWithout_Reg_Comma_PR)
        # featuresWithIdFile.write("{}: {}\n".format(key,value))

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    with open("FeaturesWithout_Reg_Comma_PRP.txt", "r", encoding = "utf_8_sig") as text:
        tempdict = json.load(text)
        finder = BigramCollocationFinder.from_documents(tempdict.values())
    print(finder.nbest(bigram_measures.raw_freq, 15))
    # amountWords = finder.word_fd.items()
    sort_amountWords = sorted(finder.word_fd.items(), key=operator.itemgetter(1))
    print(sort_amountWords)


    synonyms = {}
    lemmas = []
    for word, number in sort_amountWords:
        if number > 600:
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
    #print(sort_amountWords)

    dict_tag = dict(sort_amountWords)
    wordsWithTags = dict(pos_tag(dict_tag.keys()))
    #print(wordsWithTags)

    sort_amountWords_dict = dict(sort_amountWords)
    c = Counter()
    for word in sort_amountWords_dict:
        c.update(word)
    # print("result:")
    # print(sort_amountWords_dict)

    Hypernyms = {}
    Hyponyms = {}
    hyp = []
    for word, number in sort_amountWords_dict.items():
        if number < 600:
            hyp.clear()
            for i, j in enumerate(wn.synsets(word)):
                if i < 3:
                    x = list(chain(*[l.lemma_names() for l in j.hyponyms()]))
                    hyp.append(x)
            Hyponyms.update({word: hyp.copy()})
    print(Hyponyms)

    for key, value in Hyponyms.items():
        for lst in value:
            for word in lst:
                for word_dict, number in sort_amountWords_dict.items():
                    if number > 35:
                        if word_dict in lst:
                            sort_amountWords_dict.update({key: number})
                            break
    sorted_x = sorted(sort_amountWords_dict.items(), key=operator.itemgetter(1))
    print(sorted_x)

    # with open("TEST.txt", "w") as f:
    #     # for key,value in dict2.items():
    #     json.dump(sorted_x, f)
    #     # featuresWithIdFile.write("{}: {}\n".format(key,value))

    dict_sort_amountWords = dict(sorted_x)
    # print(dict_sort_amountWords)

    result_dict_popural_feature = {}
    alpha = 0.009
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
                      'VBN') and y != ('VBG') and y != ('RB')) and y !=('JJ') and y!=('RBR') and y != ('JJR') and y != (
                        'JJS') and y != ('RB') and y != ('RBS')]

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

    with open("Features_popular.txt", "w", encoding = "utf_8_sig") as Features_popular:
        # for key,value in dict2.items():
        json.dump(lst_features, Features_popular)
get_vector_applicant()