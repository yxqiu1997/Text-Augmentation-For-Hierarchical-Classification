import os
import collections
import numpy as np


base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, 'data')
raw_dir = os.path.join(data_dir, 'raw', 'content')
tf_idf_dir = os.path.join(data_dir, 'tf-idf')


def get_info_from_score_file(index):
    score_file = 'score-' + str(index) + '.txt'
    score_dir = os.path.join(tf_idf_dir, score_file)
    word_list = []
    tf_idf_list = []

    with open(score_dir, 'r', encoding='utf-8') as r:
        lines = r.readlines()
        for line in lines:
            tokens = line.strip().split(' ')
            if len(tokens) == 2:
                word_list.append(tokens[0])
                tf_idf_list.append(tokens[1])
            else:
                continue

    return word_list, tf_idf_list


def get_replace_prob(index):
    word_list, tf_idf_list = get_info_from_score_file(index)
    cur_tf_idf = collections.defaultdict(int)
    for i in range(len(word_list)):
        cur_tf_idf[word_list[i]] = float(tf_idf_list[i])

    replace_prob = []
    for word in word_list:
        replace_prob += [cur_tf_idf[word]]

    replace_prob = np.array(replace_prob)
    replace_prob = np.max(replace_prob) - replace_prob
    replace_prob = (replace_prob / replace_prob.sum() * len(word_list))

    return replace_prob
