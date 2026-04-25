"""use hf esm2 to get embedding"""
import torch
import numpy as np
import pandas as pd
import argparse

from torch import nn
import torch.nn.functional as F
import esm
import os, re
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel, T5EncoderModel, T5Tokenizer


class ESM_model(nn.Module):
    def __init__(self, config):
        super(ESM_model, self).__init__()
        self.model = EsmModel.from_pretrained(config.model_name)
    def forward(self, input_tokens, attention_mask):
        outputs = self.model(input_tokens, attention_mask=attention_mask).last_hidden_state
        return outputs

class T5_model(nn.Module):
    def __init__(self, config):
        super(T5_model, self).__init__()
        self.model = T5EncoderModel.from_pretrained(config.model_name)
    def forward(self, input_tokens, attention_mask):
        outputs = self.model(input_tokens, attention_mask=attention_mask).last_hidden_state
        return outputs
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Embedding extraction")
    parser.add_argument('--feat_dir', required=True, help="path to embeddings", type=str)
    parser.add_argument('--model_name', required=True, type=str, default="facebook/esm2-t33-650M-UR50D")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if "esm" in args.model_name:
        model = ESM_model(args)
    elif "t5" in args.model_name:
        model = T5_model(args)
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")
    model.to(device)
    model.eval()

    if "t5" in args.model_name:
        protein_tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    else:
        protein_tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset_list = ['esp']
    # dataset_list = ["reactzyme"]
    for dataset in dataset_list:
        print(f'Embedding extraction for: {dataset}')
        path = f'./datasets/{dataset}/'
    
        train_df = pd.read_csv(os.path.join(path, 'train.csv'))
        test_df = pd.read_csv(os.path.join(path, 'test.csv'))
        val_df = pd.read_csv(os.path.join(path, 'val.csv'))
    
        df = pd.concat([train_df, test_df, val_df], ignore_index=True)
    
        protein_set = df['Protein'].drop_duplicates().reset_index(drop=True)
        protein_index = {protein: index for index, protein in enumerate(protein_set)}
    
        df['Protein_index'] = df['Protein'].map(protein_index)
    
        
        if "t5" in args.model_name:
            dataset_dir = f'{args.feat_dir}/{dataset}_t5'
            feat_dir = os.path.join(dataset_dir, 't5')
        elif "esm" in args.model_name:
            dataset_dir = f'{args.feat_dir}/{dataset}_esm'
            feat_dir = os.path.join(dataset_dir, 'esm')
    
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            os.makedirs(feat_dir)

        for seq, index in tqdm(protein_index.items()):
            if len(seq) > 1200: # cut off with 1200 length
                seq = seq[:1200]
            if "t5" in args.model_name:
                seq = " ".join(list(re.sub(r"[UZOB]", "X", seq))) 
            inputs = protein_tokenizer(seq, return_tensors="pt").to(device)
            attention_mask = inputs.attention_mask.to(device)
            input_tokens = inputs.input_ids.to(device)
            with torch.no_grad():
                feat = model(input_tokens, attention_mask)
                feat = feat.squeeze(dim=0) # [L, dim]
                feat = feat.cpu().detach()
                torch.save(feat, os.path.join(feat_dir, f'{index}.pt'))
            
    
        train_df['Protein_index'] = train_df['Protein'].map(protein_index)
        val_df['Protein_index'] = val_df['Protein'].map(protein_index)
        test_df['Protein_index'] = test_df['Protein'].map(protein_index)
        if "t5" in args.model_name:
            train_df['Protein_Path'] = train_df['Protein_index'].apply(lambda x: f'{dataset_dir}/t5/{x}.pt')
            val_df['Protein_Path'] = val_df['Protein_index'].apply(lambda x: f'{dataset_dir}/t5/{x}.pt')
            test_df['Protein_Path'] = test_df['Protein_index'].apply(lambda x: f'{dataset_dir}/t5/{x}.pt')
        elif "esm" in args.model_name:
            train_df['Protein_Path'] = train_df['Protein_index'].apply(lambda x: f'{dataset_dir}/esm/{x}.pt')
            val_df['Protein_Path'] = val_df['Protein_index'].apply(lambda x: f'{dataset_dir}/esm/{x}.pt')
            test_df['Protein_Path'] = test_df['Protein_index'].apply(lambda x: f'{dataset_dir}/esm/{x}.pt')
        else:
            raise ValueError(f"Invalid model name: {args.model_name}")
    
        train_df.to_csv(os.path.join(dataset_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(dataset_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(dataset_dir, 'test.csv'), index=False)
