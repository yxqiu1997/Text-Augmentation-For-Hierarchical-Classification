import os
import csv
import nltk
import sys
import re
from absl import app
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


'''
    Command line usage:
        For preprocessing training set, use:
            python raw.py train
        For preprocessing testing set, use:
            python raw.py test
    
    Preprocessing pipeline:
    Paragraph Division --- Sentence Tokenization --- Convert to Lowercase --- 
    Word Tokenization --- Remove Stopwords --- POS Tagging --- Text Lemmatization --- 
'''


JOB_NAME = sys.argv[1]
BASE_DIR = os.path.dirname(__file__)
RAW_CSV_FILENAME = JOB_NAME + '.csv'
JOB_DIR = os.path.join(BASE_DIR, 'raw', JOB_NAME)
RAW_FILE = os.path.join(BASE_DIR, 'raw', RAW_CSV_FILENAME)
CONTENT_FILE = os.path.join(JOB_DIR, 'content.txt')
LABEL_FILE = os.path.join(JOB_DIR, 'label.txt')


def main(_):
    with open(CONTENT_FILE, 'w', encoding='utf-8') as w_c:
        with open(LABEL_FILE, 'w', encoding='utf-8') as w_l:
            with open(RAW_FILE, 'r', encoding='utf-8') as r:
                reader = csv.reader(r)
                index = 0
                for row in reader:
                    label_str = '__label__' + row[0]
                    w_l.write(label_str + '\n')

                    row[1] = eval(repr(row[1]).replace('\\', ' ')).strip()
                    row[2] = eval(repr(row[2]).replace('\\', ' ')).strip()
                    if len(row[1]) + len(row[2]) < 300:
                        str1 = preprocess(row[1])
                        str2 = preprocess(row[2])
                        normalised_str = str1 + str2
                    else:
                        paragraph = row[1] + row[2]
                        sentence_list = split_sentence_by_punc(paragraph)
                        normalised_str = ''
                        for sentence in sentence_list:
                            normalised_str += preprocess(sentence)
                    w_c.write(normalised_str + '\n')

                    index += 1
                    print('Finished: ' + str(index))


def split_sentence_by_punc(paragraph):
    sentence_list = []
    index = 0

    for split_punc in [".", ";", ",", " ", ""]:
        if split_punc == " " or not split_punc:
            offset = 100
        else:
            offset = 5

        while index < len(paragraph):
            if split_punc:
                pos = paragraph.find(split_punc, index + offset)
            else:
                pos = index + offset
            if pos != -1:
                sentence_list += [paragraph[index: pos + 1]]
                index = pos + 1
            else:
                sentence_list += [paragraph[index:]]
                break

    return sentence_list


def preprocess(line):
    # Tokenization
    words = tokenize_paragraph(line)

    # Remove Stopwords
    stop_words = stopwords.words('english')
    for w in ['!', ',', '.', '?', '-s', '-ly', '</s>', 's', '(', ')', '-',
              '\'s', 's\'', '#', ';', '\\', '``', '\'\'', '\'', '--', ':',
              '$', '..', '...']:
        stop_words.append(w)
    # filtered_words = [word for word in words[0] if word not in stop_words]
    filtered_words = []
    for word in words[0]:
        if word not in stop_words and not re.search(r'\d', word):
            filtered_words.append(word)

    # POS Tagging
    tags = []
    for word in filtered_words:
        tmp = [word]
        tags.append(nltk.pos_tag(tmp))

    # Lemmatization
    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for i in range(0, len(tags)):
        for tag in tags[i]:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))

    # Append return string
    new_str = ""
    for token in lemmas_sent:
        new_str += token + " "

    return new_str


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def tokenize_paragraph(line):
    words = []
    sentences = nltk.sent_tokenize(line)
    for sentence in sentences:
        words.append(nltk.word_tokenize(sentence))

    # Convert to lower case
    for i in range(0, len(words[0])):
        word = words[0][i]
        words[0][i] = str.lower(word)

    return words


if __name__ == '__main__':
    app.run(main)
