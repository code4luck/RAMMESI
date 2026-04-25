from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import torch
import pandas as pd


class FaissIndexConfig:
    def __init__(self,config):
        # Retrieval
        self.top_k = config.top_k
        self.temp = config.temp
        self.use_molecule_retrieval = config.use_molecule_retrieval
        self.gamma = config.gamma # combine p_E and p_M: p = gamma*p_E + (1-gamma)*p_M
        

        # FAISS
        self.faiss_gpu = config.faiss_gpu
        self.use_ivf = config.use_ivf
        self.nlist = config.nlist
        self.nprobe = config.nprobe  # used only for IVF
        self.index_method = config.index_method

        # Safety
        self.clamp_logit = config.clamp_logit

def get_embedding_df(df_path):
    """需要获取自定义的embedding"""
    return pd.read_csv(df_path)

def l2_normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 normalization (safe for 1D or 2D tensors)."""
    
    if x.ndim == 1:
        denom = torch.norm(x) + 1e-12
        return x / denom
    denom = torch.norm(x, dim=-1, keepdim=True) + 1e-12
    return x / denom

def to_tensor2d(x):
    # tensor reshape [D] -> [1, D]
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"to_tensor2d: x must be a tensor, got {type(x)}")
    if x.ndim == 1:
        x = x.reshape(1, -1) 
    return x

def to_numpy2d(x) -> np.ndarray:
    """Convert (D,) or (N, D) torch/np to contiguous np.float32 2D array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype="float32")
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return np.ascontiguousarray(x)


def l2_normalize_nd(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization for np arrays (N, D)."""
    den = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / den

