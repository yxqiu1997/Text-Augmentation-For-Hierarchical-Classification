import os


BASE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE_DIR, 'raw')
AUG_DIR = os.path.join(BASE_DIR, 'aug')
LABEL_DIR = os.path.join(BASE_DIR, 'label')
AUG_CONTENT = os.path.join(BASE_DIR, 'train-content.txt')
AUG_LABEL = os.path.join(BASE_DIR, 'train-label.txt')

with open(AUG_CONTENT, 'w', encoding='utf-8') as w_c:
    with open(AUG_LABEL, 'w', encoding='utf-8') as w_l:
        for index in range(0, 12):
            raw_filename = 'train-' + str(index) + '.txt'
            label_filename = 'label-' + str(index) + '.txt'
            aug_filename = 'tfidf-' + str(index) + '.txt'
            raw_file = os.path.join(RAW_DIR, raw_filename)
            label_file = os.path.join(LABEL_DIR, label_filename)
            aug_file = os.path.join(AUG_DIR, aug_filename)

            with open(raw_file, 'r', encoding='utf-8') as r_c:
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


