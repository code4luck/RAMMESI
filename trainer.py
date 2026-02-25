import torch, os, sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import seed_everything

from litmodel import LitModel
from dataset import LitDataset

from utils.metrics_utils import MetricsToFileCallback
import os
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--prot_model_locate", type=str, default="model/esm2_650m")
    parser.add_argument("--mol_model_locate", type=str, default="data/unimol_esp/esp_repr_unimolv2_dict.pt")

    # dataset
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_folder", type=str, default="./data/rxn")
    parser.add_argument("--data_name", type=str, default="rxn")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--replace_atom", action='store_true')

    parser.add_argument("--protein_col_name", type=str, default="Protein")
    parser.add_argument("--mol_col_name", type=str, default="SMILES")
    parser.add_argument("--label_col_name", type=str, default="Y")
    parser.add_argument("--protein_embed_col", type=str, default="Protein_Path")

    # model
    parser.add_argument("--unbalance", action='store_true')
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--theta", type=float, default=0.5)
    
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--proj_size", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--protein_dim", type=int, default=1280)
    parser.add_argument("--mol_in_feats", type=int, default=75)
    parser.add_argument("--mol_dim", type=int, default=256)
    parser.add_argument("--mol_num_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--num_transformer_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Trainer
    parser.add_argument("--num_labels", type=int, default=1)
    parser.add_argument("--monitor", type=str, default="val_auprc")
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=3.0)
    parser.add_argument("--gradient_clip_algorithm", type=str, default="value")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, default="32")
    args = parser.parse_args()

    seed_everything(args.seed, workers=True)
    return args

def find_last_ckpt(ckpt_dir, ckpt_name, prev_best_ckpt=None):
 
    if not os.path.exists(ckpt_dir):
        return None
    
    last_ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.exists(last_ckpt_path):
        print(f"Found Lightning default last checkpoint: {last_ckpt_path}")
        return last_ckpt_path
    if ckpt_name:
        last_ckpt_name = ckpt_name + "_last.ckpt"
        last_ckpt_path = os.path.join(ckpt_dir, last_ckpt_name)
        if os.path.exists(last_ckpt_path):
            print(f"Found custom last checkpoint: {last_ckpt_path}")
            return last_ckpt_path
        
        best_ckpt_path = os.path.join(ckpt_dir, prev_best_ckpt + ".ckpt") 
        if os.path.exists(best_ckpt_path):
             print(f"Warning: 'last.ckpt' not found. Resuming from BEST checkpoint: {best_ckpt_path}")
             print("Note: Training will restart from the epoch where the best model was saved (progress after that epoch is lost).")
             return best_ckpt_path
    
    return None

def main():
    args = parse_args()
    # load data
    data_name = args.data_name

    # get dataset and data_loader
    dataset = LitDataset(args=args)

    print("args: ", args)
    prot_model_name = args.prot_model_locate.split('/')[-1]
    if "unimol" in args.mol_model_locate:
        mol_model_name = "unimol"
    else:
        mol_model_name = args.mol_model_locate.split('/')[-1].split('.')[0]

    monitor = args.monitor

    model = LitModel(args)
    mode = "max"
    ckpt_name = f"ckpt_lr-{args.lr}_patience-{args.patience}"
    ckpt_dir = f"checkpoints/{prot_model_name}_{mol_model_name}-{data_name}"
    model_checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir, 
        monitor=monitor,
        mode=mode,
        filename=ckpt_name,
        verbose=True,
        save_weights_only=False,
        save_last=True,
    )
    early_stop = EarlyStopping(
        monitor=monitor, mode=mode, patience=args.patience, verbose=True
    )

    metrics_folder = f"results/{prot_model_name}_{mol_model_name}_{data_name}_lr-{args.lr}_patience-{args.patience}"
    metrics2file = MetricsToFileCallback(
        folder=metrics_folder,
    )


    trainer = pl.Trainer(
        # precision=args.precision,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        num_nodes=args.num_nodes,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        deterministic=True,
        sync_batchnorm=True, 
        callbacks=[model_checkpoint, early_stop, metrics2file],
        log_every_n_steps=50,
    )


    last_ckpt_path = find_last_ckpt(ckpt_dir,ckpt_name)
    print("\n--- Starting Training ---")
    start_train_time = time.time()
    if last_ckpt_path is not None and os.path.exists(last_ckpt_path):
        print(f'latest_ckpt_path detected, resuming from {last_ckpt_path}')
        trainer.fit(model, datamodule=dataset, ckpt_path=last_ckpt_path)
    else:
        print("No latest_ckpt_path found.")
        trainer.fit(model, datamodule=dataset)
    end_train_time = time.time()
    print(f"Training finished. Total training time: {end_train_time - start_train_time:.2f} seconds.")
    print(f"Batch size: {args.batch_size}, Number of batches (train): {trainer.num_training_batches}, Number of epochs: {trainer.current_epoch}")

    print("\n--- Starting Testing ---")
    import gc
    torch.cuda.empty_cache() 
    gc.collect()

    start_test_time = time.time()
    trainer.test(ckpt_path=model_checkpoint.best_model_path, datamodule=dataset)
    end_test_time = time.time()
    print(f"Testing finished. Total testing time: {end_test_time - start_test_time:.2f} seconds.")

if __name__ == "__main__":
    main()
