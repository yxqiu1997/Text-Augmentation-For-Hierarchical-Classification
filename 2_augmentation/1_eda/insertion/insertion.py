import csv
import random
from absl import app
from nltk.corpus import wordnet
from math import ceil

random.seed(1)


# Random insertion
# Randomly insert n words into the sentence


def main(_):
    with open('raw_data\\train.csv', 'r') as r:
        with open('insertion_data\\ins_aug.txt', 'w') as w:
            reader = csv.reader(r)
            num = 0
            for row in reader:
                sentence = row[1].strip().split(' ')
                insert_num = ceil(0.1 * len(sentence))
                for _ in range(insert_num):
                    add_word(sentence)

                tmp = ""
                for token in sentence:
                    tmp = tmp + token + " "
                new_row = row[0] + ", " + tmp.strip()
                w.write(new_row + "\n")

                num += 1
                print(num)


def add_word(sentence):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = sentence[random.randint(0, len(sentence) - 1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_index = random.randint(0, len(sentence) - 1)
    sentence.insert(random_index, random_synonym)


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


if __name__ == "__main__":
    app.run(main)
