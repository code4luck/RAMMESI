import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
from torch.nn.utils.rnn import pad_sequence


def _concat_padded_embeddings(
    protein_embedding: torch.Tensor,
    mol_embedding: torch.Tensor, 
    protein_mask: torch.Tensor,
    mol_mask: torch.Tensor,
    validate: bool = True
):
    """
    高效拼接已经padding的protein和molecule embeddings
    
    Args:
        protein_embedding: [B, L_p, dim] - 已padding的protein embeddings
        mol_embedding: [B, L_m, dim] - 已padding的molecule embeddings
        protein_mask: [B, L_p] - protein mask (1=valid, 0=pad)
        mol_mask: [B, L_m] - molecule mask (1=valid, 0=pad)
        validate: 是否进行输入验证
        
    Returns:
        concat_embedding: [B, L_p + L_m, dim] - concatenated embeddings
        concat_mask: [B, L_p + L_m] - concatenated mask
    """
    
    # input validation (optional, for debugging)
    if validate:
        # check batch size
        assert protein_embedding.shape[0] == mol_embedding.shape[0], \
            f"Batch size mismatch: protein {protein_embedding.shape[0]} vs mol {mol_embedding.shape[0]}"
        assert protein_mask.shape[0] == mol_mask.shape[0], \
            f"Mask batch size mismatch: protein {protein_mask.shape[0]} vs mol {mol_mask.shape[0]}"
        
        # check embedding dimension
        assert protein_embedding.shape[2] == mol_embedding.shape[2], \
            f"Embedding dimension mismatch: protein {protein_embedding.shape[2]} vs mol {mol_embedding.shape[2]}"
        
        # check mask and embedding sequence length match
        assert protein_embedding.shape[1] == protein_mask.shape[1], \
            f"Protein length mismatch: embedding {protein_embedding.shape[1]} vs mask {protein_mask.shape[1]}"
        assert mol_embedding.shape[1] == mol_mask.shape[1], \
            f"Molecule length mismatch: embedding {mol_embedding.shape[1]} vs mask {mol_mask.shape[1]}"
        
        # check mask value
        assert protein_mask.dtype in [torch.long, torch.int, torch.bool], \
            f"Protein mask should be integer or bool type, got {protein_mask.dtype}"
        assert mol_mask.dtype in [torch.long, torch.int, torch.bool], \
            f"Molecule mask should be integer or bool type, got {mol_mask.dtype}"
    
    concat_embedding = torch.cat([protein_embedding, mol_embedding], dim=1) # [B, L_p + L_m, dim]
    concat_mask = torch.cat([protein_mask, mol_mask], dim=1) # [B, L_p + L_m]
    
    return concat_embedding, concat_mask


def concat_padded_embeddings_returns(
    protein_embedding: torch.Tensor,
    mol_embedding: torch.Tensor,
    protein_mask: torch.Tensor,
    mol_mask: torch.Tensor,
    max_length: int = None,
    return_positions: bool = False,
    validate: bool = True
) -> dict:
    """
    Args:
        protein_embedding: [B, L_p, dim]
        mol_embedding: [B, L_m, dim]
        protein_mask: [B, L_p]
        mol_mask: [B, L_m]
        max_length: if specified, truncate or pad the result to this length, actually not needed, because the dataset processes the protein and does not truncate it
        return_positions: whether to return protein length information
        validate

    # if need position information, get it from the original mask
    L_p = protein_mask.shape[1]  
    L_m = mol_mask.shape[1]     

    # or get actual length (without padding)
    actual_protein_lens = protein_mask.sum(dim=1)  # [B]
    actual_mol_lens = mol_mask.sum(dim=1)          # [B]
        
    Returns:
        dict containing:
            - 'embedding': concatenated embeddings
            - 'mask': concatenated mask
            - 'protein_len': protein fixed length L_p
            - 'seq_lens': actual sequence length for each sample (optional)
    """
    
    # basic concatenation
    concat_embedding, concat_mask = _concat_padded_embeddings(
        protein_embedding, mol_embedding, 
        protein_mask, mol_mask,
        validate=validate
    )
    
    batch_size = concat_embedding.shape[0]
    current_length = concat_embedding.shape[1]
    dim = concat_embedding.shape[2]
    device = concat_embedding.device
    
    # handle max_length
    if max_length is not None:
        if current_length > max_length:
            # truncate
            concat_embedding = concat_embedding[:, :max_length]
            concat_mask = concat_mask[:, :max_length]
        elif current_length < max_length:
            # Padding
            pad_length = max_length - current_length
            embedding_pad = torch.zeros(
                batch_size, pad_length, dim, 
                dtype=concat_embedding.dtype, 
                device=device
            )
            mask_pad = torch.zeros(
                batch_size, pad_length,
                dtype=concat_mask.dtype,
                device=device
            )
            concat_embedding = torch.cat([concat_embedding, embedding_pad], dim=1)
            concat_mask = torch.cat([concat_mask, mask_pad], dim=1)
            
    return concat_embedding, concat_mask

if __name__ == "__main__":
    pass