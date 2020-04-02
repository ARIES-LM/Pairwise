#!/bin/bash
dataset=
datapath=
lamb=
modelname=


CUDA_VISIBLE_DEVICES=0 python -u main.py --model ${modelname} --vocab $datapath/vocab.new.100d.lower.pt \
--corpus $datapath/train --valid $datapath/val \
--test $datapath/test --test_order $datapath/test.order \
--writetrans decoding/${modelname}.devorder --gnnl 2 --lamb_rela ${lamb} --d_pair 256 --d_label 2 \
--batch_size 64 --beam_size 8 --senenc rnn --lr 1.0 --optimizer Adadelta --grad_clip 0.0 --seed 1234 \
--d_emb 100 --d_rnn 512 --d_mlp 512 --input_drop_ratio 0.1 --drop_ratio 0.1 --attdp 0.1 \
--save_every 2 --maximum_steps 100 --early_stop 3 >${modelname}.train 2>&1


