#!/bin/bash

mkdir models
cd models

mkdir en-de
cd en-de
wget https://s3.amazonaws.com/models.huggingface.co/bert/Helsinki-NLP/opus-mt-en-de/config.json
wget https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/pytorch_model.bin
wget https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/source.spm
wget https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/target.spm
wget https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/tokenizer_config.json
wget https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/vocab.json

cd ..

mkdir de-en
cd de-en
wget https://s3.amazonaws.com/models.huggingface.co/bert/Helsinki-NLP/opus-mt-de-en/config.json
wget https://cdn.huggingface.co/Helsinki-NLP/opus-mt-de-en/pytorch_model.bin
wget https://cdn.huggingface.co/Helsinki-NLP/opus-mt-de-en/source.spm
wget https://cdn.huggingface.co/Helsinki-NLP/opus-mt-de-en/target.spm
wget https://cdn.huggingface.co/Helsinki-NLP/opus-mt-de-en/tokenizer_config.json
wget https://cdn.huggingface.co/Helsinki-NLP/opus-mt-de-en/vocab.json