import csv
import random
from math import ceil
from absl import app
random.seed(1)


# Random swap
# Randomly swap two words in the sentence n times


def main(_):
    with open('raw_data\\train.csv', 'r') as r:
        with open('swap_data\\swap_aug.txt', 'w') as w:
            reader = csv.reader(r)
            num = 0
            for row in reader:
                raw_sentence = row[1].strip().split(' ')
                swap_num = ceil(0.1 * len(raw_sentence))
                sentence = raw_sentence.copy()
                for _ in range(swap_num):
                    sentence = swap_word(sentence)

                tmp = ""
                for token in sentence:
                    tmp = tmp + token + " "
                new_row = row[0] + ", " + tmp.strip()
                w.write(new_row + "\n")

                num += 1
                print(num)


def swap_word(sentence):
    random_index_1 = random.randint(0, len(sentence) - 1)
    random_index_2 = random_index_1
    counter = 0
    while random_index_1 == random_index_2:
        random_index_2 = random.randint(0, len(sentence) - 1)
        counter += 1
        if counter > 3:
            return sentence
    sentence[random_index_1], sentence[random_index_2] = \
        sentence[random_index_2], sentence[random_index_1]
    return sentence


if __name__ == "__main__":
    app.run(main)
