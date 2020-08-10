import os


# Activate tensorflow-gpu environment
command = "conda activate t2t"
os.system(command)

# Process the original training set file
# Split long sentence
command = "python split_by_sentence.py"
os.system(command)

# Forward translation
command = "python t2t_decoder.py \
            --problem=translate_enfr_wmt32k_rev \
            --model=transformer \
            --hparams_set=transformer_big \
            --hparams=\"sampling_method=random,sampling_temp=0.8\" \
            --decode_hparams=\"beam_size=1,batch_size=16\" \
            --checkpoint_path=checkpoints/enfr/model.ckpt-500000 \
            --output_dir=/tmp/t2t/ \
            --decode_from_file=back_translation_data/forward_src/for_src_file.txt \
            --decode_to_file=back_translation_data/forward_gen/for_gen_file.txt \
            --data_dir=checkpoints"
os.system(command)

# Back translation
command = "python t2t_decoder.py \
            --problem=translate_enfr_wmt32k_rev \
            --model=transformer \
            --hparams_set=transformer_big \
            --hparams=\"sampling_method=random,sampling_temp=0.8\" \
            --decode_hparams=\"beam_size=1,batch_size=16,alpha=0\" \
            --checkpoint_path=checkpoints/fren/model.ckpt-500000 \
            --output_dir=/tmp/t2t/ \
            --decode_from_file=back_translation_data/forward_gen/for_gen_file.txt \
            --decode_to_file=back_translation_data/backward_gen/back_gen_file.txt \
            --data_dir=checkpoints"
os.system(command)

# Compose unsupervised training set
command = "python compose_augmented_data.py"
os.system(command)
