#!/bin/bash

ori_pth=/pth/to/original/model 
ckpt_pth=./checkpoints/retro_qwen2.5-7b-instruct-1M/ppo_retro_train/global_step_190/actor
export_pth=./export_model/retro_qwen2.5-7b-instruct-1M/ppo_retro_train/global_step_190/actor

python3 verl/scripts/model_merger_1.py --backend fsdp \
    --is-value-model \
    --hf_model_path $ori_pth \
    --local_dir $ckpt_pth \
    --target_dir $export_pth

cp -L $ori_pth/tokenizer.json $export_pth
cp -L $ori_pth/merges.txt  $export_pth
cp -L $ori_pth/sparse_attention_config.json $export_pth
cp -L $ori_pth/tokenizer_config.json $export_pth
cp -L $ori_pth/vocab.json $export_pth

