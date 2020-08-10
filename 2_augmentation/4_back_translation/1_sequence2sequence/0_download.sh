#!/bin/bash

mkdir checkpoints
cd checkpoints

filename=back_trans_checkpoints.zip
wget https://storage.googleapis.com/uda_model/text/${filename}
unzip ${filename} && rm ${filename}