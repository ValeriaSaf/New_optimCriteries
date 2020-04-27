import stanza
import json

#Function marks each word with a tag of part of speech with Stanford's parser's help.
def tag_with_Stanford():
    with open('trunc_MusicInstrument.json', 'r') as f:
        jsonData = json.load(f)

    all_text_tag = []
    list_reviews = []
    for i in jsonData:
        str_i = str(i["reviewText"])
        list_reviews.append(str_i)
    print(list_reviews)

    total = []
    for i in list_reviews:
        nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', use_gpu=True, pos_batch_size=3000) # Build the pipeline, specify part-of-speech processor's batch size
        doc = nlp(i) # Run the pipeline on the input text
        wordDict = {}
        for sentence in doc.sentences:
            for word in sentence.words:
                wordDict.update({"text":word.lemma,"xpos":word.xpos,"upos":word.upos})
                all_text_tag.append(wordDict.copy())
                wordDict.clear()
        total.append(all_text_tag.copy())
        all_text_tag.clear()

    with open("Tag_list_Stanford.txt", "w") as tag_list:
            json.dump(total, tag_list,indent=4)

#Function finds name's group with Stanford's parser's help.
def deparse_name_group():
    with open('trunc_MusicInstrument.json', 'r') as f:
        jsonData = json.load(f)

    list_reviews = []
    for i in jsonData:
        str_i = str(i["reviewText"])
        list_reviews.append(str_i)

    name_group = []
    bag_of_list = []
    for i in list_reviews:
        nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
        doc = nlp(i)
        wordDict_name_group = {}
        for sentence in doc.sentences:
            for word in sentence.words:
                wordDict_name_group.update({"word-1": word.text, "word-2": sentence.words[word.head-1].text, "deprel": word.deprel })
                bag_of_list.append(wordDict_name_group.copy())
                wordDict_name_group.clear()
        for i in bag_of_list:
            for key,value in i.items():
                 if key == "deprel" and value == "compound":
                    name_group.append(i.copy())

    list_name_group = []
    inter_list = []
    for i in name_group:
        for key,values in i.items():
            if key == 'word-1' or key =='word-2':
                inter_list.append(values)
        list_name_group.append(inter_list.copy())
        inter_list.clear()

    with open("Name_group_Stanford.txt", "w") as name_group:
            json.dump(list_name_group, name_group,indent=4)

#tag_with_Stanford()
#deparse_name_group()