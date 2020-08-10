import csv
import random
from math import ceil
from random import shuffle
from absl import app
random.seed(1)


# Random deletion
# Randomly delete words from the sentence with probability p


def main(_):
    with open('raw_data\\train.csv', 'r') as r:
        with open('deletion_data\\del_aug.txt', 'w') as w:
            reader = csv.reader(r)
            num = 0
            for row in reader:
                raw_sentence = row[1].strip().split(' ')

                if len(raw_sentence) == 1:
                    w.write(row[0] + ", " + row[1] + "\n")
                else:
                    new_words = []
                    for word in raw_sentence:
                        r = random.uniform(0, 1)
                        if r > 0.4:
                            new_words.append(word)

                    if len(new_words) == 0:
                        rand_init = random.randint(0, len(raw_sentence) - 1)
                        w.write(row[0] + ", " + raw_sentence[rand_init] + "\n")
                    else:
                        tmp = ""
                        for word in new_words:
                            tmp = tmp + word + " "
                        w.write(row[0] + ", " + tmp + "\n")

                num += 1
                print(num)


if __name__ == "__main__":
    app.run(main)
