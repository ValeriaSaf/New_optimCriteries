import json
import nltk
from nltk.stem import WordNetLemmatizer
import csv
import nltk.sentiment.sentiment_analyzer
from nltk.corpus import stopwords
from string import punctuation
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from itertools import chain
import re
from numpy import *
import nltk.sentiment.sentiment_analyzer

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

#Function finds all synonyms from features. Returns dict with synonyms.
def syn():
    with open('Features_popular.txt', 'r', encoding = "utf_8_sig") as Features_popular:
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

# def syno():
#     synonyms = []
#
#     for syn in wn.synsets("chemical"):
#         for l in syn.lemmas():
#             synonyms.append(l.name())
#     print(set(synonyms))
#
#     for i, j in enumerate(wn.synsets('chemical')):
#         print("Hypernyms:", ", ".join(list(chain(*[l.lemma_names() for l in j.hypernyms()]))))
#         print("Hyponyms:", ", ".join(list(chain(*[l.lemma_names() for l in j.hyponyms()]))))
# syno()
#Function finds all hyponyms from features. Returns dict with hyponyms.
def hypo():
    with open('Features_popular.txt', 'r',encoding = "utf_8_sig") as Features_popular:
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

#Function finds all hypernyms from features. Returns dict with hypernyms.
def hype():
    with open('Features_popular.txt', 'r', encoding = "utf_8_sig") as Features_popular:
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

#Function makes tag, that defines negative context
def NegativeWord():
    with open('trunc.json', 'r', encoding = "utf_8_sig") as f:
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
    with open("Tag_nagative.txt", "w", encoding = "utf_8_sig") as tag_negative:
        json.dump(tag_negative_words, tag_negative, indent=4)

    return tag_negative_words

#Function transfers features into vector space. Output is vector consists of 0,1,-1.
def get_full_vector():
    with open('trunc.json', 'r', encoding = "utf_8_sig") as f:
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

    with open('Features_popular.txt', 'r', encoding = "utf_8_sig") as Features_popular:
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
        for k,v in sent_vec.items():
            if (k=="overall") and (v<5):
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

    return sentence_vectors
get_full_vector()