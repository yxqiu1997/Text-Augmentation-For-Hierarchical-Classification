from absl import app
from collections import Counter
import os
import math


base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, 'data')
raw_dir = os.path.join(data_dir, 'raw', 'content')
tf_idf_dir = os.path.join(data_dir, 'tf-idf')


def get_content_list():
    file_index = 0
    content_list = []

    while file_index < 12:
        file = 'train-' + str(file_index) + '.txt'
        file_dir = os.path.join(raw_dir, file)

        with open(file_dir, 'r', encoding='utf-8') as r:
            lines = r.readlines()
            contents = ""
            for i in range(len(lines)):
                contents += lines[i].strip() + ' '

            content_list.append(contents.strip())

        file_index += 1

    return content_list


def perform_tokenization(content_list):
    word_list = []
    for i in range(len(content_list)):
        word_list.append(content_list[i].split(' '))

    return word_list


def compute_word_freq(word_list):
    count_list = []
    for i in range(len(word_list)):
        count_list.append(Counter(word_list[i]))

    return count_list


def compute_tf(word, count):
    return count[word] / sum(count.values())


def num_containing(word, count_list):
    return sum(1 for count in count_list if word in count)


def compute_idf(word, count_list):
    return math.log(len(count_list) / (1 + num_containing(word, count_list)))


def compute_tf_idf(word, count, count_list):
    return compute_tf(word, count) * compute_idf(word, count_list)


def main(_):
    # Get a list containing contents (as a string) of all 12 files
    content_list = get_content_list()

    # Tokenization
    word_list = perform_tokenization(content_list)

    # Compute word freq
    count_list = compute_word_freq(word_list)

    # Print TF-IDF
    # for i, count in enumerate(count_list):
    #     print("Top words in document {}".format(i + 1))
    #     scores = {word: compute_tf_idf(word, count, count_list) for word in count}
    #     sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    #     for word, score in sorted_words[:]:
    #         print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))

    # Save TF-IDF score
    for i, count in enumerate(count_list):
        score_filename = 'score-' + str(i) + '.txt'
        score_file = os.path.join(tf_idf_dir, score_filename)
        with open(score_file, 'w', encoding='utf-8') as w:
            scores = {word: compute_tf_idf(word, count, count_list) for word in count}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            for word, score in sorted_words[:]:
                w.write(word + ' ' + str(round(score, 5)) + '\n')
                # idf = compute_idf(word, count_list)
                # w.write(word + ' ' + str(round(score, 5))
                #         + ' ' + str(round(idf, 5)) + '\n')


if __name__ == "__main__":
    app.run(main)
