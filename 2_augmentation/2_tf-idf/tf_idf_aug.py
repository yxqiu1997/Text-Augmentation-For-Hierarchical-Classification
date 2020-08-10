from absl import app
import os
import sys
import collections
from nltk.corpus import wordnet
import random
random.seed(1)


base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, 'data')
raw_dir = os.path.join(data_dir, 'raw', 'content')
tf_idf_dir = os.path.join(data_dir, 'tf-idf')
model_dir = os.path.join(base_dir, 'model')


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def main(_):
    index = sys.argv[1]
    score_file = 'score-' + str(index) + '.txt'
    score_dir = os.path.join(tf_idf_dir, score_file)
    word_list = collections.defaultdict(float)

    with open(score_dir, 'r', encoding='utf-8') as r:
        lines = r.readlines()
        for line in lines:
            content = line.strip().split(' ')
            if len(content) == 2:
                word_list[content[0]] = float(content[1])
            else:
                continue

    train_file = 'train-' + str(index) + '.txt'
    train_dir = os.path.join(raw_dir, train_file)
    aug_file = 'tfidf-' + str(index) + '.txt'
    aug_dir = os.path.join(data_dir, 'aug', aug_file)

    with open(train_dir, 'r', encoding='utf-8') as r:
        with open(aug_dir, 'w', encoding='utf-8') as w:
            lines = r.readlines()
            for line in lines:
                tokens = line.strip().split(' ')
                new_line = ""
                for token in tokens:
                    if word_list[token] >= 0.0:
                        new_line += token + ' '
                    else:
                        synonyms = get_synonyms(token)
                        if len(synonyms) >= 1:
                            synonyms = random.choice(list(synonyms))
                            new_line += synonyms + ' '
                        else:
                            new_line += token + ' '

                w.write(new_line + '\n')


if __name__ == "__main__":
    app.run(main)
