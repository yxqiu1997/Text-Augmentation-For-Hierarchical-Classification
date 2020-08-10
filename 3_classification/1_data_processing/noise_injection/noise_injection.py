import sys
import os


'''
    Command line usage:
        For deletion/insertion/substitution/swap work on character level, use:
            python noise_injection.py cDeletion/cInsertion/cSubstitution/cSwap
        respectively
        For typos caused by keyboard/ocr, use:
            python noise_injection.py cKeyboard/cOcr
        respectively
'''

JOB_NAME = sys.argv[1]
BASE_DIR = os.path.dirname(__file__)
JOB_DIR = os.path.join(BASE_DIR, JOB_NAME)
RAW_CONTENT_DIR = os.path.join(BASE_DIR, 'raw')
RAW_LABEL_DIR = os.path.join(BASE_DIR, 'label')
AUG_CONTENT = os.path.join(BASE_DIR, JOB_NAME, 'train-content.txt')
AUG_LABEL = os.path.join(BASE_DIR, JOB_NAME, 'train-label.txt')

with open(AUG_CONTENT, 'w', encoding='utf-8') as w_c:
    with open(AUG_LABEL, 'w', encoding='utf-8') as w_l:
        for index in range(0, 12):
            aug_filename = JOB_NAME + '-' + str(index) + '.txt'
            raw_content = 'train-' + str(index) + '.txt'
            raw_label = 'label-' + str(index) + '.txt'
            aug_file = os.path.join(JOB_DIR, aug_filename)
            content_file = os.path.join(RAW_CONTENT_DIR, raw_content)
            label_file = os.path.join(RAW_LABEL_DIR, raw_label)

            with open(content_file, 'r', encoding='utf-8') as r_c:
                with open(label_file, 'r', encoding='utf-8') as r_l:
                    contents = r_c.readlines()
                    labels = r_l.readlines()
                    for content in contents:
                        w_c.write(content.strip() + '\n')
                    for label in labels:
                        w_l.write(label.strip() + '\n')

            with open(aug_file, 'r', encoding='utf-8') as r_c:
                with open(label_file, 'r', encoding='utf-8') as r_l:
                    contents = r_c.readlines()
                    labels = r_l.readlines()
                    for content in contents:
                        w_c.write(content.strip() + '\n')
                    for label in labels:
                        w_l.write(label.strip() + '\n')
