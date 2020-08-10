import os


BASE_DIR = os.path.dirname(__file__)
GPT2 = os.path.join(BASE_DIR, 'gpt2', 'train-content.txt')
XLNET = os.path.join(BASE_DIR, 'xlnet', 'train-content.txt')

GPT2_SUB1 = os.path.join(BASE_DIR, 'gpt2', 'train-content-1.txt')
GPT2_SUB2 = os.path.join(BASE_DIR, 'gpt2', 'train-content-2.txt')

XLNET_SUB1 = os.path.join(BASE_DIR, 'xlnet', 'train-content-1.txt')
XLNET_SUB2 = os.path.join(BASE_DIR, 'xlnet', 'train-content-2.txt')

with open(GPT2, 'w', encoding='utf-8') as w:
    with open(GPT2_SUB1, 'r', encoding='utf-8') as r:
        lines = r.readlines()
        for line in lines:
            w.write(line.strip() + '\n')
    with open(GPT2_SUB2, 'r', encoding='utf-8') as r:
        lines = r.readlines()
        for line in lines:
            w.write(lines.strip() + '\n')

with open(XLNET, 'w', encoding='utf-8') as w:
    with open(XLNET_SUB1, 'r', encoding='utf-8') as r:
        lines = r.readlines()
        for line in lines:
            w.write(line.strip() + '\n')
    with open(XLNET_SUB2, 'r', encoding='utf-8') as r:
        lines = r.readlines()
        for line in lines:
            w.write(lines.strip() + '\n')
