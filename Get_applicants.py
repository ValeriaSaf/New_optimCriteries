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
from numpy import *
import json
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.tag import pos_tag
from nltk.chunk import api
from nltk.chunk.api import ChunkParserI
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import stanza

stemmer = SnowballStemmer('english')

#function of WordNet lemmatizer
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

#Function gets word or phrase applicant of feature. Use tokens,lemmatizer,tag. Make chunk and particular gramma.
def get_word_applicant():
    with open('trunc.json', 'r', encoding = "utf_8_sig") as f:
        jsonData = json.load(f)

    file_handler = open("Features.txt", "w", encoding = "utf_8_sig")
    dict1 = {}
    for i in jsonData:
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
    print(dict1)
    file_handler.close()
    return t

#------------------------------------------------Algorithm_on_base_Stanford's_parser------------------------------------

#Function gets word or phrase applicant of feature. Use tokens,lemmatizer,tag. Make chunk and particular gramma with Stanford's parser's help.
def get_word_application_Stanford():
    with open('Tag_list_Stanford.txt', 'r', encoding = "utf_8_sig") as f:
        jsonData = json.load(f)

    list_token = []
    full_token_sent = []
    for i in jsonData:
        for st in i:
            for key,value in st.items():
                    if key=="text":
                        list_token.append(value)
        full_token_sent.append(list_token.copy())
        list_token.clear()
    #print(full_token_sent)

    file_handler = open("Features_Stanford.txt", "w", encoding = "utf_8_sig")
    list_application_stanford = []
    count = 0
    for i in full_token_sent:
        pos_text = pos_tag(i)
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
            count += 1
            if subtree.label() == 'Chunk':
                t = subtree
                t = ' '.join(word for word, pos in t.leaves())
                file_handler.write(str(t) + "\n")
        list_application_stanford.append(pos_text)
    file_handler.close()


#Function clears list of applicants with Stanford's parser's help.
def clear_tag_Stanford():
    with open('Features_Stanford.txt', 'r',encoding = "utf_8_sig") as Features_popular:
        features_text = Features_popular.readlines()


    all_text = []
    total = []
    for i in features_text:
        nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', use_gpu=True, pos_batch_size=3000) # Build the pipeline, specify part-of-speech processor's batch size
        doc = nlp(i) # Run the pipeline on the input text
        wordDict = {}
        for sentence in doc.sentences:
            for word in sentence.words:
                wordDict.update({"text":word.lemma,"xpos":word.xpos,"upos":word.upos})
                all_text.append(wordDict.copy())
                wordDict.clear()
        total.append(all_text.copy())
        all_text.clear()
        print(total)

    file_handler = open("Clear_Features_Stanford.txt", "w", encoding = "utf_8_sig")
    for i in total:
        for lst in i:
            del_lst = dict(lst)
            for key,value in del_lst.items():
                if (key=="xpos" and value=="AFX") or (key=="xpos" and value=="DT") or (key=="xpos" and value=="VB") or \
                        (key=="xpos" and value=="RB") or (key=="xpos" and value=="MD") or (key=="xpos" and value=="PRP") or (key=="xpos" and value==","):
                 lst.clear()
    file_handler.write(str(total) + "\n")
    file_handler.close()
    #print(total)

def list_clear_feature():

   #  with open('Clear_Features_Stanford.txt', 'r') as Features_popular:
   #      features_text = Features_popular.read()
   #  new_str = features_text.replace("'",'"')
   #  new_str1 = new_str.replace('"""','"\'"')
   #  new_str3 = new_str1.replace('""','"')
   #  new_str4 = new_str3.replace('mic"ing', 'DEL')
   #  new_str2 = new_str1.replace('"text":  "','"text": DEL')
   # # new_str = re.sub(r"']", "DEL", new_str)
   #
   #  with open("Clear_Features_Stanford.txt", "w") as name_groups:
   #      name_groups.write(new_str4)
   #  name_groups.close()


    with open('Clear_Features_Stanford.txt', 'r', encoding = "utf_8_sig") as clear_features_popular:
        clear_feature_json = json.load(clear_features_popular)

    for i in clear_feature_json:
        for st in i:
            del_lst = dict(st)
            for key,value in del_lst.items():
                if (key=="xpos" and value=="LS") or (key=="upos" and value=="PUNCT") or \
                        (key=="xpos" and value=="CC") or (key=="xpos" and value=="ADD") or (key=="xpos" and value=="UH") or \
                        (key=="xpos" and value=="VBP") or (key=="xpos" and value=="CD"):
                    st.clear()
    print(clear_feature_json)

    list_clear = []
    full_list = []
    for i in clear_feature_json:
        for st in i:
            #list_clear.append(st["text"])
            for k,v in st.items():
                if k=="text":
                    list_clear.append(v)
        full_list.append(list_clear.copy())
        list_clear.clear()
    print(full_list)

    with open("Features_Stanford.txt", "w", encoding = "utf_8_sig") as name_groups:
        file_text = ""
        for lst in full_list:
            for word in lst:
                file_text += word+" "
            if (len(lst) > 0):
                file_text += "\n"
        name_groups.write(file_text)

get_word_applicant()
#get_word_application_Stanford()
#clear_tag_Stanford()
#list_clear_feature()