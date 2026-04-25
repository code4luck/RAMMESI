#!/bin/bash

# use two gpus with accumulate_grad_batches=2, batch_size=64
# use 4 gpus with accumulate_grad_batches=1, batch_size=64

# export CUDA_VISIBLE_DEVICES=0,1
python trainer.py \
  --devices 2 \
  --accumulate_grad_batches 2 \
  --accelerator gpu \
  --strategy ddp \
  --batch_size 64 \
  --max_epochs 50 \
  --num_workers 4 \
  --data_folder ./datasets/esp \
  --data_name esp_unimol_esm_IFL \
  --lr 3e-5 \
  --patience 50 \
  --monitor val_auroc \
  --num_labels 1 \
  --mol_model_locate ./mol_repr/unimol_esp/esp_mol_repr.pt \
  --prot_model_locate ./model/esm2_650m \
  --seed 3407 \
  --temp 1.1 \
  --proj_size 768 \
  --hidden_size 768 \
  --protein_dim 1280 \
  --mol_dim 1536 \
  --num_transformer_layers 3 \
  --dropout 0.1 \
  --protein_col_name Protein \
  --mol_col_name SMILES \
  --label_col_name Y \
  --max_length 1200 \
  --protein_embed_col Protein_Path \
  --unbalance \
  --gamma 3 \
  --theta 0.4 \
  --alpha 0.65