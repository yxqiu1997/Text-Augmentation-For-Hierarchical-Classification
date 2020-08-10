import sys
import os


'''
    Command line usage:
        For deletion/insertion/swap/synonyms work, use:
            python eda.py deletion/insertion/swap/synonyms
        respectively
'''

JOB_NAME = sys.argv[1]
BASE_DIR = os.path.dirname(__file__)
JOB_DIR = os.path.join(BASE_DIR, JOB_NAME)
AUG_FILE = os.path.join(JOB_DIR, 'aug.txt')
AUG_CONTENT = os.path.join(JOB_DIR, 'train-content.txt')
AUG_LABEL = os.path.join(JOB_DIR, 'train-label.txt')
RAW_CONTENT = os.path.join(BASE_DIR, 'raw', 'content.txt')
RAW_LABEL = os.path.join(BASE_DIR, 'raw', 'label.txt')

with open(AUG_CONTENT, 'w', encoding='utf-8') as w_c:
    with open(AUG_LABEL, 'w', encoding='utf-8') as w_l:
        with open(RAW_CONTENT, 'r', encoding='utf-8') as r_c:
            with open(RAW_LABEL, 'r', encoding='utf-8') as r_l:
                with open(AUG_FILE, 'r', encoding='utf-8') as r_a:
                    # Process augmentation document
                    lines = r_a.readlines()
                    for line in lines:
                        cols = line.split(',')
                        w_l.write(cols[0].strip() + '\n')
                        w_c.write(cols[1].strip() + '\n')
                    # Process raw document
                    contents = r_c.readlines()
                    labels = r_l.readlines()
                    for content in contents:
                        w_c.write(content.strip() + '\n')
                    for label in labels:
                        w_l.write(label.strip() + '\n')
