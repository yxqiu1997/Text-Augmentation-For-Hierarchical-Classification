import os


BASE_DIR = os.path.dirname(__file__)
T5_DIR = os.path.join(BASE_DIR, 't5')
RAW_DIR = os.path.join(BASE_DIR, 'raw')
LABEL_DIR = os.path.join(BASE_DIR, 'label')
CONTENT_FILE = os.path.join(T5_DIR, 'train-content.txt')
LABEL_FILE = os.path.join(T5_DIR, 'train-label.txt')

with open(CONTENT_FILE, 'w', encoding='utf-8') as w_c:
    with open(LABEL_FILE, 'w', encoding='utf-8') as w_l:
        for index in range(0, 12):
            raw_content_filename = 'train-' + str(index) + '.txt'
            raw_label_filename = 'label-' + str(index) + '.txt'
            aug_filename = 'aug-' + str(index) + '.txt'
            raw_content_file = os.path.join(RAW_DIR, raw_content_filename)
            raw_label_file = os.path.join(LABEL_DIR, raw_label_filename)
            aug_file = os.path.join(T5_DIR, aug_filename)

            with open(raw_content_file, 'r', encoding='utf-8') as r_c:
                with open(raw_label_file, 'r', encoding='utf-8') as r_l:
                    with open(aug_file, 'r', encoding='utf-8') as r_a:
                        contents = r_c.readlines()
                        labels = r_l.readlines()
                        augs = r_a.readlines()
                        for content in contents:
                            w_c.write(content.strip() + '\n')
                        for aug in augs:
                            w_c.write(aug.strip() + '\n')
                        for i in range(0, 2):
                            for label in labels:
                                w_l.write(label.strip() + '\n')
