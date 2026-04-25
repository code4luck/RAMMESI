"""
- Keep FAISS strictly for retrieval.
- Maintain two independent indices:
  1) Enzyme index built from CLEAN (or similar) embeddings.
  2) Molecule index built from ECFP / learned molecule vectors.
- During inference, you can:
  - Retrieve top-k enzyme neighbors for E_q.
  - Retrieve top-k molecule neighbors for M_q (optional switch).
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import faiss
import torch
import faiss.contrib.torch_utils
from faiss_utils import l2_normalize_tensor, to_tensor2d, to_numpy2d, l2_normalize_nd

class DualIndexRAG:
    """
    Minimal dual-index retriever with model-driven pseudo labeling.
    """

    def __init__(
        self,
        dim_enzyme,
        dim_mol,
        cfg,
        predict_on_enzyme_neighbors=None, #  predict on enzyme neighbors
        predict_on_molecule_neighbors=None, #  predict on molecule neighbors
    ):
        self.cfg = cfg
        self.dim_enzyme = dim_enzyme
        self.dim_mol = dim_mol if dim_mol is not None else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # FAISS indices + metadata
        self.enz_index = None
        self.mol_index = None
        self.enz_keys: Optional[List[str]] = None
        self.mol_keys: Optional[List[str]] = None
        # GPU resource (if any)
        self._gpu_res = None
     
    def _make_index_cpu(self, dim):
        # set index method  l2/ ip no need to train
        if self.cfg.index_method == "flat_l2":
            index = faiss.IndexFlatL2(dim) # l2 distance
        elif self.cfg.index_method == "flat_ip":
            index = faiss.IndexFlatIP(dim) # inner product
        elif self.cfg.index_method == "ivfflat":
            quantizer = faiss.IndexFlatIP(dim) # coarse quantizer, build cluster
            # default use inner product
            index = faiss.IndexIVFFlat(quantizer, dim, self.cfg.nlist,faiss.METRIC_INNER_PRODUCT)
        elif self.cfg.index_method == "ivfpq":
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, self.cfg.nlist, self.cfg.m_pq, 8)
        else:
            raise ValueError(f"Invalid index method: {self.cfg.index_method}")
        return index

    def index_to_gpu(self, index):
        """
        gpu cpu management
        """
        if self.cfg.faiss_gpu and self.device.type == "cuda":
            try:
                if self._gpu_res is None:
                    self._gpu_res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(self._gpu_res, 0, index)
                if hasattr(index, "nprobe"):
                    try:
                        index.nprobe = int(self.cfg.nprobe)
                    except Exception:
                        print("gpu use nprobe is not supported")
                return index, True
            except Exception:
                if hasattr(index, "nprobe"):
                    try:
                        index.nprobe = int(self.cfg.nprobe)
                    except Exception:
                        print("cpu use nprobe is not supported")
                        
                print("!!!!!want to move index to gpu, but failed!!!!")
        return index, False

    def _train_index(self, index, xb):
        # only for the ivfflat and ivfpq
        if self.cfg.index_method == "ivfflat" or self.cfg.index_method == "ivfpq":
            if hasattr(index, "is_trained") and not index.is_trained:
                index.train(xb) # train on cpu
        return index

    def _prepare_db(self, X, dim):
        """
        convert data to np [N, dim] 
        normalize data if needed
        """
        xb = to_numpy2d(X)  # (N, D) 
        if self.cfg.index_method in ["flat_ip", "ivfflat", "ivfpq"]:
            xb = l2_normalize_nd(xb)
        if xb.shape[1] != dim:
            raise ValueError(f"Dim mismatch: got {xb.shape[1]}, expected {dim}")
        return xb
    
    def _build_common(self, X, dim):
        """
            build index adjust device
            存在numpy 和 tensor的两种类型的输入调整
        """
        
        # (N, D) _prepare_db  input will be converted to numpy type and placed on cpu
        xb_np = self._prepare_db(X, dim)  
        # CPU index           
        cpu_index = self._make_index_cpu(dim)  
        # train IVF if needed
        cpu_index = self._train_index(cpu_index, xb_np)  
        # move to GPU if enabled
        index, is_on_gpu = self.index_to_gpu(cpu_index) 
        if is_on_gpu and isinstance(X, torch.Tensor) and X.is_cuda:
            # CUDA Tensor input to GPU index
            x_add = to_tensor2d(X)
            if self.cfg.index_method in ["flat_ip", "ivfflat", "ivfpq"]:
                x_add = l2_normalize_tensor(x_add)
            x_add = x_add.contiguous().to(torch.float32)
            ids_t = torch.arange(x_add.size(0), device=x_add.device, dtype=torch.int64)
            index.add_with_ids(x_add, ids_t)
        else:
            # CPU index or input is not CUDA Tensor: use NumPy
            ids = np.arange(xb_np.shape[0], dtype=np.int64)
            try:
                index.add_with_ids(np.ascontiguousarray(xb_np), ids)
            except Exception:
                index.add(np.ascontiguousarray(xb_np))
        return index, is_on_gpu     
    
    def build_enzyme_index(self, enz_embs, enz_keys):
        """
            1.enz_embs torch.Tensor(sim embedding) 
            2.enz_keys List[str] seqs
        """
        assert enz_embs.ndim == 2 and enz_embs.shape[1] == self.dim_enzyme, f" build_enzyme_index: enz_embs.shape: {enz_embs.shape}, dim_enzyme: {self.dim_enzyme}"
        assert len(enz_keys) == enz_embs.shape[0], f" build_enzyme_index: len(enz_keys): {len(enz_keys)}, enz_embs.shape[0]: {enz_embs.shape[0]}"

        self.enz_index, self._enz_on_gpu = self._build_common(enz_embs, self.dim_enzyme) # build index
        self.enz_keys = list(enz_keys)
        
    def build_molecule_index(self, mol_vecs, mol_keys):
        """
            1.mol_vecs torch.Tensor(sim embedding) 
            2.mol_keys List[str] smiles
        """
        assert self.dim_mol is not None, "dim_mol must be set to build molecule index"
        assert mol_vecs.ndim == 2 and mol_vecs.shape[1] == self.dim_mol, f" build_molecule_index: mol_vecs.shape: {mol_vecs.shape}, dim_mol: {self.dim_mol}"
        assert len(mol_keys) == mol_vecs.shape[0], f" build_molecule_index: len(mol_keys): {len(mol_keys)}, mol_vecs.shape[0]: {mol_vecs.shape[0]}"

        self.mol_index, self._mol_on_gpu= self._build_common(mol_vecs, self.dim_mol)
        self.mol_keys = list(mol_keys)
        
    # ---------- Search ----------
    def _prep_queries(self, Q, is_on_gpu=False):
        """
            index check if cuda can be used
            query [B, dim]
        """
        if isinstance(Q, torch.Tensor):
            if Q.ndim == 1: Q = Q[None, :] # [B,] -> [B, 1]
            if self.cfg.index_method in ["flat_ip", "ivfflat", "ivfpq"]:
                Q = l2_normalize_tensor(Q).contiguous()
            if is_on_gpu:
                return Q.cuda() if Q.device.type != "cuda" else Q
            else: # index not on gpu, convert to numpy and query
                return to_numpy2d(Q.cpu())
    
        Qn = to_numpy2d(Q)  # [B, D]
        if self.cfg.index_method in ["flat_ip", "ivfflat", "ivfpq"]:
            Qn = l2_normalize_nd(Qn)
        return Qn # np.ndarray [B, D]
    
    def _search(self, index, Q, k, is_on_gpu=False):
        Qn = self._prep_queries(Q, is_on_gpu) # convert to [B, dim]
        D, I = index.search(Qn, k)  # D/I: (B, k)
        return D, I

    def _search_single(self, index, q, k, is_on_gpu=False):
        D, I = self._search(index, q, k, is_on_gpu)
        return D[0], I[0]


    def search_enzyme(self, q_emb, top_k=None, exclude_key=None):
        """
        single query retrieval
        q_embed tensor [B, dim] is best
        exclude_key dont need
        K = int(top_k or self.cfg.top_k)  use the passed top_k
        """
        # q_emb tensor
        assert self.enz_index is not None, "Enzyme index not built"
        K = int(top_k or self.cfg.top_k)
        scores, idxs = self._search_single(self.enz_index, q_emb, K + 1, self._enz_on_gpu)
        cands = []
        for s, idx in zip(scores, idxs):
            if idx < 0:
                continue
            row = int(idx)
            key = self.enz_keys[row] if self.enz_keys is not None else str(row)
            if exclude_key is not None and key == exclude_key: # ! dont use
                continue 
            cands.append({"row": row, "key": key, "score": float(s)}) # idx, seq, sim_score
        return cands[:K]
    
    def search_molecule(self, q_vec, top_k=None, exclude_key=None):
        """
        single query retrieval [dim,]
        """
        
        # q_vec tensor -> recommend cpu
        assert self.mol_index is not None, "Molecule index not built"
        K = int(top_k or self.cfg.top_k)
        scores, idxs = self._search_single(self.mol_index, q_vec, K + 1, self._mol_on_gpu)
        cands = []
        for s, idx in zip(scores, idxs):
            if idx < 0:
                continue
            row = int(idx)
            key = self.mol_keys[row] if self.mol_keys is not None else str(row)
            if exclude_key is not None and key == exclude_key: 
                continue 
            cands.append({"row": row, "key": key, "score": float(s)})
        return cands[:K]

    def search_enzyme_batch(self, Q_emb, top_k=None, exclude_keys=None):
        """
        batch query retrieval [B, dim]
        K = int(top_k or self.cfg.top_k)  use the passed top_k
        """
        # Q_emb CLEAN embedding [B, dim] -> cpu is best
        assert self.enz_index is not None, "Enzyme index not built"
        K = int(top_k or self.cfg.top_k) 
        D, I = self._search(self.enz_index, Q_emb, K + 1, self._enz_on_gpu)  # (B, K+1)
        
        B = D.shape[0]
        if exclude_keys is not None and len(exclude_keys) != B: # we dont use
            raise ValueError("exclude_keys must be None or a list of length B")
        out = []
        
        # get top k for each query
        for b in range(B):
            scores_b, idxs_b = D[b], I[b]
            ex_key = (exclude_keys[b] if (exclude_keys is not None) else None)
            cands_b = []
            for s, idx in zip(scores_b, idxs_b):
                if idx < 0:
                    continue
                row = int(idx)
                key = self.enz_keys[row] if self.enz_keys is not None else str(row)
                if ex_key is not None and key == ex_key:
                    continue
                cands_b.append({"row": row, "key": key, "score": float(s)})
            out.append(cands_b[:K])
        return out

    def search_molecule_batch(self, Q_vec, top_k=None, exclude_keys=None):
        # K = int(top_k or self.cfg.top_k)  use the passed top_k
        # Q_vec ECFP/learned molecule vectors [B, dim] -> cpu is best
        assert self.mol_index is not None, "Molecule index not built"
        K = int(top_k or self.cfg.top_k)
        D, I = self._search(self.mol_index, Q_vec, K + 1, self._mol_on_gpu)  # (B, K+1)
        B = D.shape[0]
        if exclude_keys is not None and len(exclude_keys) != B:
            raise ValueError("exclude_keys must be None or a list of length B")
        
        # get queried data
        out = []
        for b in range(B):
            scores_b, idxs_b = D[b], I[b]
            ex_key = (exclude_keys[b] if (exclude_keys is not None) else None)
            cands_b: List[Dict] = []
            for s, idx in zip(scores_b, idxs_b):
                if idx < 0:
                    continue
                row = int(idx)
                key = self.mol_keys[row] if self.mol_keys is not None else str(row)
                if ex_key is not None and key == ex_key:
                    continue
                cands_b.append({"row": row, "key": key, "score": float(s)})
            out.append(cands_b[:K])
        return out