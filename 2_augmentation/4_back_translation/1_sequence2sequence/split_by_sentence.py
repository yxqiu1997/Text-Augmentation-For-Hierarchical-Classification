import tensorflow as tf
import nltk
from absl import app
import json


def main(_):
    tf.compat.v1.logging.info("****** split_by_sentence.py ******");

    # Load original training set
    with tf.io.gfile.GFile("original_data/original_content.csv", "r") as inf:
        contents = inf.readlines()

    # Divide data
    contents = contents[0: len(contents)]

    # Split by sentence
    new_contents = []
    doc_len = []

    # decode characters which are not in utf-8
    for i in range(len(contents)):
        contents[i] = contents[i].strip()
        if isinstance(contents[i], bytes):
            contents[i] = contents[i].decode("utf-8")
        sentence_list = nltk.tokenize.sent_tokenize(contents[i])

        sentence_list = split_paragraph(sentence_list)
        doc_len += [len(sentence_list)]
        for sentence in sentence_list:
            new_contents += [sentence]

    # Store new training contents
    with tf.io.gfile.GFile("back_translation_data/forward_src/for_src_file.txt", "w") as ouf:
        for sentence in new_contents:
            ouf.write(sentence + "\n")

    with tf.io.gfile.GFile("back_translation_data/doc_len/doc_len_file.json", "w") as ouf:
        json.dump(doc_len, ouf)


def split_paragraph(sentence_list):
    new_contents = []

    for punc in [",", ".", ";", " ", ""]:
        if punc == " " or not punc:
            offset = 100
        else:
            offset = 5
        long_sentence_flag = False

        for sentence in sentence_list:
            if len(sentence) < 300:
                new_contents += [sentence]
            else:
                long_sentence_flag = True
                split_sentence = split_by_punc(sentence, punc, offset)
                new_contents += split_sentence

        if not long_sentence_flag:
            break

    return new_contents


def split_by_punc(sentence, punc, offset):
    sentence_list = []
    index = 0

    while index < len(sentence):
        if punc:
            pos = sentence.find(punc, index + offset)
        else:
            pos = index + offset
        if pos != -1:
            sentence_list += [sentence[index: pos + 1]]
            index = pos + 1
        else:
            sentence_list += [sentence[index:]]
            break

    return sentence_list


if __name__ == "__main__":
    app.run(main)