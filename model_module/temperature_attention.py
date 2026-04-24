import torch, math
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
import warnings

class TempAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, temp=1.0, dropout=0.1, bias=True, kdim=None, vdim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.gamma = nn.Parameter(torch.tensor(temp, requires_grad=True))
        self.gamma_min = 0.7
        self.gamma_max = 2.0
        self.dropout = dropout
        
        if not self._qkv_same_embed_dim:
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        else:
            self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()
    
    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj.weight)
            if self.in_proj.bias is not None:
                nn.init.constant_(self.in_proj.bias, 0.)
        else:
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            if self.q_proj.bias is not None:
                nn.init.constant_(self.q_proj.bias, 0.)
            if self.k_proj.bias is not None:
                nn.init.constant_(self.k_proj.bias, 0.)
            if self.v_proj.bias is not None:
                nn.init.constant_(self.v_proj.bias, 0.)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(self, query, key, value, key_padding_mask=None, batch_protein_lens=None, batch_mol_lens=None, max_protein_len=None, need_weights=False):
        """
        for key_padding_mask [B, L]
        the pad is 0 and the unpad is 1
        """
        batch_size, seq_len, embed_dim = query.shape

        if self._qkv_same_embed_dim:
            qkv = self.in_proj(query)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)      
    
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # get the attn score
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # [B, n-heads, Lq, Lk]
        # use temperature factor for the cross-modal attention
        clamped_gamma = torch.clamp(self.gamma, min=self.gamma_min, max=self.gamma_max)
        
        # use for loop
        # batch temperature attention opt
        for i in range(batch_size):
            protein_len_i = batch_protein_lens[i].item()
            mol_len_i = batch_mol_lens[i].item()
            # Apply temperature scaling to protein -> mol (protein-to-molecule)
            scores[i, :, :protein_len_i, max_protein_len:max_protein_len+mol_len_i] *= clamped_gamma
            # Apply temperature scaling to mol -> protein (molecule-to-protein)
            scores[i, :, max_protein_len:max_protein_len+mol_len_i, :protein_len_i] *= clamped_gamma
        
        #########################################################
        # [L, L]  Vectorized temperature scaling
        """
        B, H, Lq, Lk = scores.shape
        assert Lq == Lk, "Lq and Lk must be equal in concat setting"
        row_indices = torch.arange(Lq, device=scores.device, dtype=torch.long)[None, None, :, None]  # [1, 1, Lq, 1]
        col_indices = torch.arange(Lk, device=scores.device, dtype=torch.long)[None, None, None, :]  # [1, 1, 1, Lk]
        pl = batch_protein_lens[:, None, None, None]  # [B, 1, 1, 1]
        ml = batch_mol_lens[:, None, None, None]      # [B, 1, 1, 1]
        max_pl = torch.full_like(pl, max_protein_len)  # Broadcast global max_Lp to [B,1,1,1]
        # cross_mask_pm Protein|mol --> protein 长度 行内的 + protein列外的 + mol列内的 -> pro->mol 区域
        cross_mask_pm = (row_indices < pl) & (col_indices >= max_pl) & (col_indices < max_pl + ml)
        # cross_mask_mp Protein|mol --> protein 长度 行外的 + prot+mol 行内的 + pro 列内的 --> mol->pro 区域
        cross_mask_mp = (row_indices >= max_pl) & (row_indices < max_pl + ml) & (col_indices < pl)
        # 1 true; 0 pad values--> 取or获取所有的 true values
        cross_mask = cross_mask_pm | cross_mask_mp # [B, 1, L, L]
        cross_mask = cross_mask.expand(-1, H, -1, -1).float() # [B, H, L, L]
        scores = scores * (1 + (clamped_gamma - 1) * cross_mask) # [B, H, L, L]
        """
        
        # use padding mask      
        if key_padding_mask is not None:
            # [B, L] -> [B, 1, 1, L]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            # pad is 0 and unpad is 1
            scores = scores.masked_fill(key_padding_mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        ) # [B, L, dim]
        output = self.out_proj(attn_output) # [B, L, dim]
        if need_weights:
            return output, attn_weights
        else:
            return output, None