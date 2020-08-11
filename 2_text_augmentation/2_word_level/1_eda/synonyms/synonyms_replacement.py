import csv
import random
from math import ceil
from random import shuffle
from absl import app
from nltk.corpus import wordnet
random.seed(1)


# Synonym Replacement (SR):
# Randomly choose n words from the sentence that are not stop words.
# Replace each of these words with one of its synonyms chosen at random.


def main(_):
    with open('raw_data\\train.csv', 'r') as r:
        with open('synonyms_data\\syn_aug.txt', 'w') as w:
            reader = csv.reader(r)
            num = 0
            for row in reader:
                raw_sentence = row[1].strip().split(' ')
                sentence = raw_sentence.copy()
                replace_limit = ceil(0.1 * len(raw_sentence))
                num_replaced = 0

                random.shuffle(raw_sentence)
                for word in raw_sentence:
                    synonyms = get_synonyms(word)
                    if len(synonyms) >= 1:
                        synonym = random.choice(list(synonyms))
                        for i in range(0, len(sentence)):
                            if sentence[i] == word:
                                sentence[i] = synonym
                        num_replaced += 1
                    if num_replaced > replace_limit:
                        break

                    tmp = ""
                    for token in sentence:
                        tmp = tmp + token + " "
                new_row = row[0] + ", " + tmp.strip()
                w.write(new_row + "\n")

                num += 1
                print(num)


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
