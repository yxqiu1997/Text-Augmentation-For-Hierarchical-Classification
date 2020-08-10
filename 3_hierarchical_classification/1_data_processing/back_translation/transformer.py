import os


BASE_DIR = os.path.dirname(__file__)
RAW_LABEL_DIR = os.path.join(BASE_DIR, 'label')
RAW_CONTENT_DIR = os.path.join(BASE_DIR, 'raw')
TRANSFORMER_DIR = os.path.join(BASE_DIR, 'transformer')
TRANSFORMER_LABEL_DIR = os.path.join(TRANSFORMER_DIR, 'label')
AUG_CONTENT = os.path.join(TRANSFORMER_DIR, 'train-content.txt')
AUG_LABEL = os.path.join(TRANSFORMER_DIR, 'train-label.txt')

with open(AUG_CONTENT, 'w', encoding='utf-8') as w_c:
    with open(AUG_LABEL, 'w', encoding='utf-8') as w_l:
        for index in range(0, 12):
            content_filename = 'train-' + str(index) + '.txt'
            label_filename = 'label-' + str(index) + '.txt'
            content_file = os.path.join(RAW_CONTENT_DIR, content_filename)
            label_file = os.path.join(RAW_LABEL_DIR, label_filename)
            with open(content_file, 'r', encoding='utf-8') as r_c:
                with open(label_file, 'r', encoding='utf-8') as r_l:
                    contents = r_c.readlines()
                    labels = r_l.readlines()
                    for content in contents:
                        w_c.write(content.strip() + '\n')
                    for label in labels:
                        w_l.write(label.strip() + '\n')

        for index in range(0, 11):
            content_filename = 'aug-' + str(index) + '.txt'
            label_filename = 'label-' + str(index) + '.txt'
            content_file = os.path.join(TRANSFORMER_DIR, content_filename)
            label_file = os.path.join(TRANSFORMER_LABEL_DIR, label_filename)
            with open(content_file, 'r', encoding='utf-8') as r_c:
                with open(label_file, 'r', encoding='utf-8') as r_l:
                    contents = r_c.readlines()
                    labels = r_l.readlines()
                    for content in contents:
                        w_c.write(content.strip() + '\n')
                    for label in labels:
                        w_l.write(label.strip() + '\n')
