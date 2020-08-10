import json

import tensorflow as tf
from absl import app


def main(_):
    translated_file = "back_translation_data/backward_gen/back_gen_file.txt"
    doc_len_file = "back_translation_data/doc_len/doc_len_file.json"
    original_content = "original_data/original_content.csv"
    original_label = "original_data/original_label.csv"

    unsupervised_augmented_file = "unsupervised_data/unsup_aug.txt"
    unsupervised_file = "unsupervised_data/unsup.txt"

    supervised_content = "supervised_data/sup_content.txt"
    supervised_label = "supervised_data/sup_label.txt"

    # Read translated file
    with tf.gfile.GFile(translated_file) as inf:
        translated_contents = inf.readlines()
    with tf.gfile.GFile(doc_len_file) as inf:
        translated_doc_len = json.load(inf)

    # Compose for unsupervised training
    cnt = 0
    with tf.gfile.GFile(unsupervised_augmented_file, "w") as ouf:
        for i, sentence_num in enumerate(translated_doc_len):
            paragraph = ""
            for _ in range(sentence_num):
                paragraph += translated_contents[cnt].strip() + " "
                cnt += 1
            ouf.write(paragraph.strip() + "\n")

    compose_content(original_content, unsupervised_augmented_file, unsupervised_file)

    # Compose for supervised training
    compose_label(original_label, supervised_label)
    compose_content(original_content, unsupervised_augmented_file, supervised_content)


def compose_label(original_label, supervised_label):
    with tf.gfile.GFile(original_label, "r") as inf:
        label_inf = inf.readlines()

    cnt = 0
    with tf.gfile.GFile(supervised_label, "w") as ouf:
        while cnt < 2:
            write_file(label_inf, ouf)
            cnt += 1


def compose_content(original_content, unsupervised_augmented_file, file):
    with tf.gfile.GFile(original_content, "r") as inf:
        ori_inf = inf.readlines()

    with tf.gfile.GFile(unsupervised_augmented_file, "r") as inf:
        aug_inf = inf.readlines()

    with tf.gfile.GFile(file, "w") as ouf:
        write_file(ori_inf, ouf)
        write_file(aug_inf, ouf)


def write_file(inf, ouf):
    for i in range(len(inf)):
        paragraph = inf[i].strip() + " "
        ouf.write(paragraph.strip() + "\n")


if __name__ == "__main__":
    app.run(main)
