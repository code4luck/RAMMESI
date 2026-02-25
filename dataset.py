import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from transformers import AutoTokenizer
from typing import List, Optional
from pytorch_lightning import LightningDataModule


class SeqMolDataset(Dataset):
    def __init__(self, df, protein_col, mol_col, label_col, protein_embed_col=None):
        self.df = df
        self.protein_seqs = df[protein_col].values.tolist()
        self.mol_smiles = df[mol_col].values.tolist()
        self.labels = df[label_col].values.tolist()
        self.protein_embed_list = df[protein_embed_col].values.tolist()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.protein_seqs[idx], self.mol_smiles[idx], self.labels[idx], self.protein_embed_list[idx]


class LitDataset(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # config
        self.mol_model_locate = args.mol_model_locate
        self.prot_model_locate = args.prot_model_locate
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # col name
        self.mol_col_name = args.mol_col_name
        self.protein_col_name = args.protein_col_name
        self.label_col_name = args.label_col_name
        self.protein_embed_col=args.protein_embed_col 
        # load tokenizer dont need tokenizer
        self.mol_tokenizer =  torch.load(self.mol_model_locate)        
        # dataset path
        self.train_data_path = args.data_folder + "/train.csv"
        self.test_data_path = args.data_folder + "/test.csv"
        self.val_data_path = args.data_folder + "/val.csv"

    def setup(self, stage: str):
        if stage == "fit":
            self.train_df = pd.read_csv(self.train_data_path)
            self.val_df = pd.read_csv(self.val_data_path)
            self.train_dataset = SeqMolDataset(df=self.train_df, protein_col=self.protein_col_name,
                                             mol_col=self.mol_col_name, label_col=self.label_col_name, protein_embed_col=self.protein_embed_col)
            
            self.val_dataset = SeqMolDataset(df=self.val_df, protein_col=self.protein_col_name,
                                             mol_col=self.mol_col_name, label_col=self.label_col_name, protein_embed_col=self.protein_embed_col)
            
        elif stage == "validate":
            self.val_df = pd.read_csv(self.val_data_path)
            self.val_dataset = SeqMolDataset(df=self.val_df, protein_col=self.protein_col_name,
                                             mol_col=self.mol_col_name, label_col=self.label_col_name, protein_embed_col=self.protein_embed_col)
            
        elif stage =="test":
            self.test_df = pd.read_csv(self.test_data_path)
            self.test_dataset = SeqMolDataset(df=self.test_df, protein_col=self.protein_col_name,
                                             mol_col=self.mol_col_name, label_col=self.label_col_name, protein_embed_col=self.protein_embed_col)


    def pad_embedding(self, embedding_list):
        """
        embedding_list: List[Tensor,]
            embedding: [L, dim]
            pad embedding list 
            dont use for graph data
        """
        max_protein_length = max(i.size(0) for i in embedding_list)
        embedding_dim = embedding_list[0].size(1)
        batch_size = len(embedding_list)

        protein_masks = torch.zeros(batch_size,max_protein_length, dtype=torch.long)
        padded_embeddings = torch.zeros(batch_size, max_protein_length, embedding_dim)

        for idx, emb in enumerate(embedding_list):
            original_len = emb.size(0)
            # Copy the original embedding into the padded tensor
            padded_embeddings[idx, :original_len, :] = emb
            # Set the corresponding part of the attention mask to 1; pad is 0 , value is 1
            protein_masks[idx, :original_len] = 1
        
        return padded_embeddings, protein_masks 

    def collate_fn(self, batch):
        protein_seqs, mol_seqs, labels, protein_embed_paths = zip(*batch)
    
        labels = torch.tensor(labels) # [B]
        protein_token_ids = None
        protein_attention_mask=None
        protein_embedding = []
        protein_embedding_mask=None
        if not self.protein_embed_col and self.protein_tokenizer:
            protein_inputs = self.protein_tokenizer(protein_seqs, padding=True, max_length=self.max_length, 
                                                        truncation=True if self.max_length is not None else False, return_tensors="pt")
            protein_token_ids = protein_inputs.input_ids
            protein_attention_mask = protein_inputs.attention_mask
        else:
            for embed_path in protein_embed_paths:
                embedding = torch.load(embed_path)
                protein_embedding.append(embedding.float())
            # pad
            protein_embedding, protein_embedding_mask = self.pad_embedding(protein_embedding)

        mol_tokens=None
        mol_attention_mask=None
        mol_embeddings = []
        mol_embedding_mask=None

        for seq in mol_seqs:
            mol_embedding = self.mol_tokenizer[seq]
            mol_embedding = mol_embedding["atomic_reprs"]
            mol_embeddings.append(torch.from_numpy(mol_embedding).float()) # List[torch.Tensor]
        mol_embeddings, mol_embedding_mask = self.pad_embedding(mol_embeddings)

        batch = {
                "protein_seq": protein_seqs,
                "protein_token_ids": protein_token_ids,
                "protein_attention_mask": protein_attention_mask,
                "protein_embeddings": protein_embedding,
                "protein_embedding_mask": protein_embedding_mask,

                "mol_seqs": mol_seqs,
                "mol_embeddings": mol_embeddings,
                "mol_embedding_mask": mol_embedding_mask,
                "mol_input": mol_tokens,
                "mol_attention_mask": mol_attention_mask,
                "labels": labels,
            }        
        return batch

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.collate_fn)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.collate_fn)



if __name__ == "__main__":
    pass
