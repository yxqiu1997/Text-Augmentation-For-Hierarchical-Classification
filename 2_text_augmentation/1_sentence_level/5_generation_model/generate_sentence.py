from absl import app
import os
import nlpaug.augmenter.sentence as nas
import sys


base_dir = os.path.dirname(__file__)
raw_dir = os.path.join(base_dir, 'data', 'raw')
aug_sentence_dir = os.path.join(base_dir, 'data', 'aug_sentence')
gpt2_aug_dir = os.path.join(aug_sentence_dir, 'GPT2')
xlnet_aug_dir = os.path.join(aug_sentence_dir, 'XLNet')
label_aug_dir = os.path.join(aug_sentence_dir, 'label')


def main(_):
    # Triple every label
    # for index in range(0, 12):
    #     label_filename = 'label-' + str(index) + '.txt'
    #     label_dir = os.path.join(raw_dir, 'label', label_filename)
    #     label_aug_file = os.path.join(label_aug_dir, label_filename)
    #
    #     with open(label_dir, 'r', encoding='utf-8') as r:
    #         with open(label_aug_file, 'w', encoding='utf-8') as w:
    #             lines = r.readlines()
    #             for line in lines:
    #                 for i in range(0, 3):
    #                     w.write(line.strip() + '\n')

    index = sys.argv[1]

    # Create sentences by XLNet
    aug = nas.ContextualWordEmbsForSentenceAug(model_path='xlnet-base-cased')

    train_filename = 'train-' + str(index) + '.txt'
    train_dir = os.path.join(raw_dir, 'content', train_filename)
    aug_filename = 'aug-' + str(index) + '.txt'
    aug_dir = os.path.join(xlnet_aug_dir, aug_filename)

    with open(train_dir, 'r', encoding='utf-8') as r:
        with open(aug_dir, 'w', encoding='utf-8') as w:
            lines = r.readlines()
            for line in lines:
                augmented_texts = aug.augment(line.strip(), n=3)
                for i in range(len(augmented_texts)):
                    w.write(augmented_texts[i].strip() + '\n')

    # Create sentences by gpt-2
    train_filename = 'train-' + str(index) + '.txt'
    train_dir = os.path.join(raw_dir, 'content', train_filename)
    aug_filename = 'aug-' + str(index) + '.txt'
    aug_dir = os.path.join(gpt2_aug_dir, aug_filename)

    with open(train_dir, 'r', encoding='utf-8') as r:
        with open(aug_dir, 'w', encoding='utf-8') as w:
            lines = r.readlines()
            for line in lines:
                for i in range(0, 3):
                    aug = nas.ContextualWordEmbsForSentenceAug(model_path='gpt2')
                    w.write(aug.augment(line.strip()).strip() + '\n')


if __name__ == '__main__':
    app.run(main)
