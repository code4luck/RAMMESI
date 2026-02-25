import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.embedding_pad_utils import concat_padded_embeddings_returns
from model_module.transformer_encoder_layer import TransformerEncoderLayer
from model_module.pooling_head import MeanPoolingHead, Attention1dPoolingHead

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, query, key_value, query_mask, key_mask):
        """
        query: [B, Lq, H], key_value: [B, Lk, H]
        query_mask/key_mask: [B, L] with 1 for valid, 0 for pad
        """
        key_padding_mask = ~key_mask.bool()
        attn_output, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(query + self.dropout(attn_output))
        x = self.norm2(x + self.ffn(x)) # [B, L_q, H]
        return x


class Gate(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_p = nn.Linear(hidden_size, hidden_size)
        self.W_c = nn.Linear(hidden_size, hidden_size)
        self.W_m = nn.Linear(hidden_size, hidden_size)
        self.W_g = nn.Linear(3 * hidden_size, 3 * hidden_size)

    def forward(self, feat_p, feat_c, feat_m):
        p = self.W_p(feat_p)
        c = self.W_c(feat_c)
        m = self.W_m(feat_m)
        gates_in = torch.cat([p, c, m], dim=1)  # [B, 3H]
        logits = self.W_g(gates_in).view(-1, 3, self.hidden_size)  # [B, 3, H]
        weights = torch.softmax(logits, dim=1)

        fused = weights[:, 0, :] * p + weights[:, 1, :] * c + weights[:, 2, :] * m
        return fused

class PMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj_size = config.proj_size
        self.hidden_size = config.hidden_size
        self.protein_dim = config.protein_dim
        self.mol_dim = config.mol_dim
        self.drop_prob = config.dropout
        self.num_attention_heads = config.num_attention_heads
        self.num_transformer_layers = config.num_transformer_layers
        self.num_labels = config.num_labels
        self.temp = config.temp
        # __init__(self, in_feats=75, dim_embedding=128, num_layers=4, padding=True, hidden_feats=None)
        self.protein_proj = nn.Sequential(
            nn.LayerNorm(self.protein_dim),
            nn.Linear(self.protein_dim, self.proj_size),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.proj_size, self.hidden_size)
        )
        self.mol_proj = nn.Sequential(
            nn.LayerNorm(self.mol_dim),
            nn.Linear(self.mol_dim, self.proj_size),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.proj_size, self.hidden_size)
        ) 

        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=self.hidden_size,
                    nhead=self.num_attention_heads,
                    dim_feedforward=2 * self.hidden_size, 
                    dropout=self.drop_prob,
                    temp=self.temp
                )
                for _ in range(self.num_transformer_layers)
            ]
        )
        # cross-attention
        self.crossattn_p2m = CrossAttentionBlock(self.hidden_size, self.num_attention_heads, self.drop_prob)
        self.crossattn_m2p = CrossAttentionBlock(self.hidden_size, self.num_attention_heads, self.drop_prob)

        # pooling
        self.mol_features = Attention1dPoolingHead(hidden_size=self.hidden_size)
        self.protein_features = Attention1dPoolingHead(hidden_size=self.hidden_size)
        self.mean_pooling = MeanPoolingHead(hidden_size=self.hidden_size)
        # self.bn_fusion = nn.BatchNorm1d(self.hidden_size)
        self.fuse_ln_p = nn.LayerNorm(self.hidden_size)
        self.fuse_ln_c = nn.LayerNorm(self.hidden_size)
        self.fuse_ln_m = nn.LayerNorm(self.hidden_size)

        self.gate = Gate(self.hidden_size)
        self.output = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.hidden_size, self.num_labels)
        )

    def forward(self, protein_embeddings, protein_embedding_mask, mol_embeddings, mol_embedding_mask, protein_token_ids=None, protein_attention_mask=None):
        """
        protein_embeddings,  [B, L, dim]
        mol_embeddings [B, L, dim]
        protein_embedding_mask, mol_embedding_mask [B, L], 1 is true values, 0 is pad values
        """
        protein_embeddings = self.protein_proj(protein_embeddings)
        mol_embeddings = self.mol_proj(mol_embeddings)
        
        max_protein_len = protein_embedding_mask.shape[1]
        batch_protein_lens = protein_embedding_mask.sum(dim=1) # [B, ]
        batch_mol_lens = mol_embedding_mask.sum(dim=1) # [B,] current batch mol

        # concat protein and mol embeddings
        concat_embedding, concat_mask = concat_padded_embeddings_returns(
            protein_embedding=protein_embeddings, mol_embedding=mol_embeddings, protein_mask=protein_embedding_mask, mol_mask=mol_embedding_mask
        )
        # ctx with conv pooling
        # forward(self, query, key_value, query_mask, key_mask)
        # [B, L_p, H]
        protein_ctx = self.crossattn_p2m(query=protein_embeddings, key_value=mol_embeddings, query_mask=protein_embedding_mask, key_mask=mol_embedding_mask)
        mol_ctx = self.crossattn_m2p(mol_embeddings, protein_embeddings, mol_embedding_mask, protein_embedding_mask) # [B, L_m, H]
        protein_feats = self.protein_features(protein_ctx, protein_embedding_mask)
        mol_feats = self.mol_features(mol_ctx, mol_embedding_mask)
        # transformer encoder
        # protein_len = protein_embedding_mask.shape[1] # L
        for layer in self.transformer_layers:
            concat_embedding = layer(src=concat_embedding, batch_protein_lens=batch_protein_lens,
                                            batch_mol_lens=batch_mol_lens, src_key_padding_mask=concat_mask, max_protein_len=max_protein_len) # [B, L, dim]
        concat_features = self.mean_pooling(concat_embedding, concat_mask)

        x = self.gate(self.fuse_ln_p(protein_feats), self.fuse_ln_c(concat_features), self.fuse_ln_m(mol_feats))
        # fc
        x = self.output(x) # [B, 1]
        return x.squeeze(-1)
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--protein_model_locate", type=str, default="esm1b_t33_650M_UR50S")
    parser.add_argument("--mol_model_locate", type=str, default="unimol_v2")
    parser.add_argument("--ckpt_weight_path", type=str, default=None)
    parser.add_argument("--hidden_size", type=int, default=1280)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_attention_heads", type=int, default=16)
    parser.add_argument("--num_transformer_layers", type=int, default=12)
    parser.add_argument("--num_labels", type=int, default=1)
    parser.add_argument("--proj_size", type=int, default=1024)
    parser.add_argument("--protein_dim", type=int, default=1280)
    parser.add_argument("--mol_dim", type=int, default=1280)
    parser.add_argument("--device", type=str, default="cuda")

    config = parser.parse_args()
    model = PMModel(config)

    protein_token_ids = torch.randint(0, 21, (4, 17))
    protein_attention_mask = torch.randint(0, 2, (4, 17))
    mol_inputs = torch.randint(0, 21, (4, 17))
    mol_attention_mask = torch.randint(0, 2, (4, 17))
    mol_seq = ["C1CCCCC1", "C1CCCCC1", "C1CCCCC1", "C1CCCCC1"]

    output = model(protein_token_ids, protein_attention_mask, mol_inputs, mol_attention_mask, mol_seq)
    print(output.shape)
