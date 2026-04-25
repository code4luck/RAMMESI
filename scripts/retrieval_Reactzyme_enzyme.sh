#!/bin/bash
set -euo pipefail

TOPKS=(5 10)
BSS=(16 16)

if [ "${#TOPKS[@]}" -ne "${#BSS[@]}" ]; then
  echo "TOPKS 和 BSS 长度不一致" >&2
  exit 1
fi

for i in "${!TOPKS[@]}"; do
  topk="${TOPKS[$i]}"
  bs="${BSS[$i]}"

  out_dir="reactzyme_IFL_K2theta04_enyzme"
  out_file="reactzyme_IFL_K2theta04_enzyme_${topk}_bs${bs}.csv"

  echo "Running enzyme-retrieval topk=${topk}, batch_size=${bs} -> ${out_dir}/${out_file}"

  python retrieval/retrieval_augmentation.py \
    --data_name reactzyme \
    --eval_file_path ./datasets/reactzyme/test.csv \
    --output_dir "${out_dir}" \
    --output_file "${out_file}" \
    --batch_size "${bs}" \
    --top_k "${topk}" \
    --protein_train_data_path ./datasets/reactzyme/train.csv \
    --checkpoint_path ./ckpt/reactzyme_esm2_unimol_IFL/ckpt_lr-3e-05_patience-50.ckpt \
    --mol_model_locate ./mol_repr/unimol_reactzyme/reactzyme_mol_repr.pt \
    --clean_embedding_path ./clean_embedding/clean_reactzyme/reactzyme_test_embedding.pt \
    --protein_target_embedding_file_path ./clean_embedding/clean_reactzyme/reactzyme_train_embedding.pt
done