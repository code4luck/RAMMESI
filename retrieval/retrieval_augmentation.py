"""
Retrieval-Augmented Inference based on FAISS
"""

import torch
from torch import nn
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
from pytorch_lightning import seed_everything
import torch.nn.functional as F
from litmodel import LitModel
from pathlib import Path
import os
from faiss_retrieval import DualIndexRAG

import time
from collections import defaultdict

PROFILE_SEC = defaultdict(float)
PROFILE_CNT = defaultdict(int)

def _sync_cuda(device: str):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

#================================ utils ========================================
def get_embedding_df(df_path):
    """get the embedding from the csv file"""
    return pd.read_csv(df_path)

Retrieval_config = {
    'top_k': 5,
    'enhancement_weight': 0.2,
    'similarity_threshold': 0.3
}

# print("Retrieval_config: ", Retrieval_config)
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

#================================ dataset ========================================

class SeqMolDataset(Dataset):
    def __init__(self, df, query_embed_path, protein_col, mol_col, label_col, protein_embed_col=None):
        self.df = df
        self.protein_seqs = df[protein_col].values.tolist()
        self.mol_smiles = df[mol_col].values.tolist()
        self.labels = df[label_col].values.tolist()
        self.protein_embed_list = df[protein_embed_col].values.tolist()
        self.query_embed = torch.load(query_embed_path)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        protein_seq = self.protein_seqs[idx]
        query_embedding = self.query_embed[protein_seq]
        return protein_seq, query_embedding, self.mol_smiles[idx], self.labels[idx], self.protein_embed_list[idx]

class EvalDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mol_model_locate = args.mol_model_locate
        self.prot_model_locate = args.prot_model_locate
        self.clean_embedding_path  = args.clean_embedding_path # ! 获取当前query的 query_embedding for CLEAN
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # col name
        self.mol_col_name = args.mol_col_name
        self.protein_col_name = args.protein_col_name
        self.label_col_name = args.label_col_name
        self.protein_embed_col=args.protein_embed_col # ! new
        # load tokenizer
        if self.protein_embed_col:
            self.protein_tokenizer = None
        else:
            self.protein_tokenizer = AutoTokenizer.from_pretrained(self.prot_model_locate)

        self.mol_tokenizer =  torch.load(self.mol_model_locate)  
        # ! eval file path
        self.eval_file_path = args.eval_file_path
        self.load_eval_data()


    def load_eval_data(self):
        self.eval_df = pd.read_csv(self.eval_file_path)

    def get_testloader(self):
        self.test_dataset = SeqMolDataset(df=self.eval_df, query_embed_path=self.clean_embedding_path, protein_col=self.protein_col_name,
                                             mol_col=self.mol_col_name, label_col=self.label_col_name, protein_embed_col=self.protein_embed_col)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)
        return self.test_loader


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
        protein_seqs, query_embeddings, mol_seqs, labels, protein_embed_paths = zip(*batch)
    
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

        # ! note the query_embedding (CLEAN) is List[tensor,] -> [B, dim] 
        query_embeddings = torch.stack(query_embeddings, dim=0)
    
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
                "query_embeddings": query_embeddings,
                "protein_token_ids": protein_token_ids, # None
                "protein_attention_mask": protein_attention_mask, # None
                "protein_embedings": protein_embedding,
                "protein_embedding_mask": protein_embedding_mask,

                "mol_seqs": mol_seqs,
                "mol_embeddings": mol_embeddings,
                "mol_embedding_mask": mol_embedding_mask,
                "mol_input": mol_tokens, # None
                "mol_attention_mask": mol_attention_mask, # None
                "labels": labels,
            }        
        return batch


#================================ inference ========================================

def pad_embedding(embedding_list):
    """
    Pad a list of embeddings to the same length.
    """
    max_len = max(i.size(0) for i in embedding_list)
    embedding_dim = embedding_list[0].size(1)
    batch_size = len(embedding_list)

    padded_masks = torch.zeros(batch_size, max_len, dtype=torch.long)
    padded_embeddings = torch.zeros(batch_size, max_len, embedding_dim)
    for idx, emb in enumerate(embedding_list):
        original_len = emb.size(0)
        padded_embeddings[idx, :original_len, :] = emb
        padded_masks[idx, :original_len] = 1
        
    return padded_embeddings, padded_masks

def _prepare_retrieval_batch(batch_retriever_seqs, batch_single_size, query_mol_seq, mol_tokenizer, protein_seq_to_path_dict):
    """
    efficiently build the batch for computing retrieval signal
    use the dictionary to get the protein embedding O(1)
    batch_retriever_seqs: retrieved sequences
    """
    mol_embeddings = []
    # copy the mol embedding to match the retrieved protein number
    for mol_seq, cur_size in zip(query_mol_seq, batch_single_size):
        if cur_size > 0:
            mol_emb_2d = torch.from_numpy(mol_tokenizer[mol_seq]["atomic_reprs"]).float()  # [L, D]
            # copy the mol embedding to match the retrieved protein number
            mol_embeddings.extend([mol_emb_2d] * cur_size)

    # if no retrieved result, return empty tensor
    if not mol_embeddings:
        return {}
    mol_embeddings, mol_embedding_mask = pad_embedding(mol_embeddings)

    # get the protein embedding
    protein_embeddings = []
    flat_retrieved_seqs = [seq for seq_list in batch_retriever_seqs for seq in seq_list]
    
    # !
    t_io0 = time.perf_counter()
    
    for seq in flat_retrieved_seqs:
        embedding_path = protein_seq_to_path_dict[seq] # O(1) dictionary lookup
        embedding = torch.load(embedding_path)
        protein_embeddings.append(embedding.float())
    # ! 
    PROFILE_SEC["protein_io_sec"] += time.perf_counter() - t_io0
    PROFILE_CNT["protein_io_loads"] += len(flat_retrieved_seqs)
    
    protein_embeddings, protein_embedding_mask = pad_embedding(protein_embeddings)

    assert protein_embeddings.shape[0] == mol_embeddings.shape[0], "Protein and molecule batch sizes for retrieval do not match"
    
    return {
        "protein_embedings": protein_embeddings,
        "protein_embedding_mask": protein_embedding_mask,
        "mol_embeddings": mol_embeddings,
        "mol_embedding_mask": mol_embedding_mask,
    }

def _compute_retrieval_batch_signal(model, device, query_mol_seq, similar_sequences, mol_tokenizer, protein_seq_to_path_dict):
    """
    compute the retrieval signal (pseudo label) for a batch.
    use the dictionary `protein_seq_to_path_dict` to improve efficiency.
    similar_sequences: [[sim_seq1, sim_seq2], [],...]
    """
    batch_single_size = [len(item) for item in similar_sequences]
    total_size = sum(batch_single_size) # cur batch queried sim seqs
    
    if total_size == 0:
        return torch.tensor([], device=device)

    try:
        with torch.no_grad():
            similar_batch = _prepare_retrieval_batch(
                batch_retriever_seqs=similar_sequences,
                batch_single_size=batch_single_size,
                query_mol_seq=query_mol_seq,
                mol_tokenizer=mol_tokenizer,
                protein_seq_to_path_dict=protein_seq_to_path_dict
            )
            # !
            _sync_cuda(device)
            t_fwd0 = time.perf_counter()
            pseudo_logits = model(
                protein_embeddings=similar_batch["protein_embedings"].to(device),
                protein_embedding_mask=similar_batch["protein_embedding_mask"].to(device),
                mol_embeddings=similar_batch["mol_embeddings"].to(device),
                mol_embedding_mask=similar_batch["mol_embedding_mask"].to(device)
            )
            pseudo_confidence = torch.sigmoid(pseudo_logits)
            
            # ! 
            _sync_cuda(device)
            PROFILE_SEC["pseudo_forward_sec"] += time.perf_counter() - t_fwd0
            PROFILE_CNT["num_pseudo_samples"] += int(total_size)
            return pseudo_confidence

    except Exception as e:
        print(f"Failed to compute retrieval signal: {e}")
        return torch.full((total_size,), 0.5, device=device)

def enhanced_logit(base_confidence, retrieval_signal):
    """compute the enhanced logits based on the confidence"""
    enhancement_weight = Retrieval_config["enhancement_weight"]
    enhanced_confidence = (1 - enhancement_weight) * base_confidence + enhancement_weight * retrieval_signal
    # convert back to logits space, add epsilon to prevent log(0)
    enhanced_logits = torch.log(enhanced_confidence / (1 - enhanced_confidence + 1e-8))
    return enhanced_logits

def _apply_batch_retrieval_enhancement(model, base_logits, base_confidence, protein_query_embeddings, protein_seqs, mol_seqs,
                                      searcher, device, mol_tokenizer, protein_seq_to_path_dict):
    """
    the core function of applying retrieval enhancement.
    use a initialized `searcher` object and dictionary mapping, no need to initialize each time
    """
    try:
        # 1. use faiss searcher to search sim embedding seqs
        # search protein-->the embedding is the CLEAN embedding
        # ! add time
        t_search0 = time.perf_counter()
        
        retrieved_results = searcher.search_enzyme_batch(Q_emb=protein_query_embeddings, top_k=Retrieval_config["top_k"])
        # ! 
        PROFILE_SEC["faiss_search_sec"] += time.perf_counter() - t_search0
        PROFILE_CNT["num_queries"] += len(protein_seqs)


        # get every query seq's retrieved seqs and scores
        batch_retriever_seqs = [[item["key"] for item in res] for res in retrieved_results]
        batch_retriever_scores = [[item["score"] for item in res] for res in retrieved_results]
        
        # ! 
        PROFILE_CNT["num_neighbors"] += sum(len(x) for x in batch_retriever_seqs)
        
        for re_seq, re_sc in zip(batch_retriever_seqs, batch_retriever_scores):
            assert len(re_seq) == len(re_sc), "batch retrieved seq not equal score"

        # 2. compute the pseudo label (retrieval signal) for all retrieved results
        retrieval_signal_flat = _compute_retrieval_batch_signal(
            model=model, device=device, query_mol_seq=mol_seqs, similar_sequences=batch_retriever_seqs,
            mol_tokenizer=mol_tokenizer, protein_seq_to_path_dict=protein_seq_to_path_dict
        ) # return pseudo_confidence
        
        if retrieval_signal_flat.numel() == 0:
            return base_logits

        # 3. aggregate the retrieval signal for each original sample and enhance it
        
        # !
        t_aggr0 = time.perf_counter()
        enhanced_logits_list = []
        current_idx = 0
        for i in range(len(protein_seqs)):
            num_retrieved = len(batch_retriever_seqs[i])
            if num_retrieved == 0:
                enhanced_logits_list.append(base_logits[i])
                continue

            # get the retrieved results and scores for the current sample
            end_idx = current_idx + num_retrieved
            retrieval_signal_cur = retrieval_signal_flat[current_idx:end_idx]
            retriever_scores_cur = torch.tensor(batch_retriever_scores[i], device=device)
            current_idx = end_idx

            # filter the low similarity results
            high_sim_mask = retriever_scores_cur > Retrieval_config["similarity_threshold"]
            if high_sim_mask.sum() > 0:
                if searcher.cfg.index_method.endswith("l2"): # when using l2 distance, need to reverse the scores
                    scores_for_weights = -retriever_scores_cur[high_sim_mask]
                else:
                    scores_for_weights = retriever_scores_cur[high_sim_mask]
                # use softmax to convert the high similarity scores to weights
                weights = F.softmax(scores_for_weights, dim=0)
                # weighted average to get the final retrieval signal
                final_retrieval_signal = torch.sum(weights * retrieval_signal_cur[high_sim_mask])
            else:
                # if no high similarity scores, use the average of all retrieved results
                final_retrieval_signal = retrieval_signal_cur.mean()
            
            # enhance the prediction
            enhanced_prediction = enhanced_logit(base_confidence[i], final_retrieval_signal)
            enhanced_logits_list.append(enhanced_prediction)
        # ! 
        PROFILE_SEC["aggregation_sec"] += time.perf_counter() - t_aggr0
        return torch.stack(enhanced_logits_list)

    except Exception as e:
        print(f"Failed to apply retrieval enhancement: {e}")
        return base_logits


def infer(args, model, eval_loader, mol_tokenizer, searcher, protein_seq_to_path_dict, mol_seq_to_path_dict=None):
    """
    execute the complete inference process.
    MODIFIED: 接收`searcher`和`protein_seq_to_path_dict`。
    """
    device = args.device
    
    # ! 
    total_samples = 0
    time_base_sec = 0.0   # only base forward
    time_retrieval_sec = 0.0  # retrieval + enhancement (delta)
    sync = lambda: torch.cuda.synchronize() if device == "cuda" and torch.cuda.is_available() else None

    PROFILE_SEC.clear()
    PROFILE_CNT.clear()

    all_results = []
    
    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            protein_query_embeddings = batch["query_embeddings"] # use cpu [B, dim]
            protein_embeddings = batch["protein_embedings"].to(device)
            protein_embedding_mask = batch["protein_embedding_mask"].to(device)
            mol_embeddings = batch["mol_embeddings"].to(device)
            mol_embedding_mask = batch["mol_embedding_mask"].to(device)
            labels = batch["labels"]

            # ---- timing: base prediction ----
            B = labels.shape[0]
            sync()
            t0 = time.perf_counter()
            # base prediction
            base_logits = model(
                protein_embeddings=protein_embeddings, protein_embedding_mask=protein_embedding_mask,
                mol_embeddings=mol_embeddings, mol_embedding_mask=mol_embedding_mask
            )
            base_confidence = torch.sigmoid(base_logits) # [B,]

            sync()
            t1 = time.perf_counter()
            time_base_sec += (t1 - t0)

            # ---- timing: retrieval enhancement (delta) ----
            sync()
            t2 = time.perf_counter()           
            
            enhanced_logits = _apply_batch_retrieval_enhancement(
                model=model,
                base_logits=base_logits,
                base_confidence=base_confidence,
                protein_query_embeddings=protein_query_embeddings,
                protein_seqs=batch["protein_seq"],
                mol_seqs=batch["mol_seqs"],
                searcher=searcher,
                device=device,
                mol_tokenizer=mol_tokenizer,
                protein_seq_to_path_dict=protein_seq_to_path_dict
            )
            enhanced_confidence = torch.sigmoid(enhanced_logits)

            sync()
            t3 = time.perf_counter()
            time_retrieval_sec += (t3 - t2)
            total_samples += B

            # save the results
            for i in range(labels.shape[0]):
                all_results.append({
                    "label": labels[i].item(),
                    "base_confidence": base_confidence[i].item(),
                    "enhanced_confidence": enhanced_confidence[i].item()
                })

    # ---------- summarize and print delta ----------
    if total_samples > 0:
        compute_only_ms = (PROFILE_SEC["faiss_search_sec"] + PROFILE_SEC["pseudo_forward_sec"]) / total_samples * 1000
        io_ms = PROFILE_SEC["protein_io_sec"] / total_samples * 1000
        aggr_ms = PROFILE_SEC["aggregation_sec"] / total_samples * 1000

        print("[Compute-only] per-sample search+pseudo = %.3f ms" % compute_only_ms)
        print("[Breakdown] per-sample faiss_search = %.3f ms" % (PROFILE_SEC["faiss_search_sec"]/total_samples*1000))
        print("[Breakdown] per-sample pseudo_forward = %.3f ms" % (PROFILE_SEC["pseudo_forward_sec"]/total_samples*1000))
        print("[Breakdown] per-sample aggregation = %.3f ms" % aggr_ms)
        print("[IO] per-sample protein torch.load = %.3f ms" % io_ms)

        if PROFILE_CNT["num_neighbors"] > 0:
            per_neighbor_ms = (PROFILE_SEC["pseudo_forward_sec"]) / PROFILE_CNT["num_neighbors"] * 1000
            print("[Pseudo] per-neighbor pseudo_forward ~= %.3f ms" % per_neighbor_ms)  
            
        t_base_per_sample_ms = (time_base_sec / total_samples) * 1000
        t_retrieval_per_sample_ms = (time_retrieval_sec / total_samples) * 1000  # delta time (ms/sample)
        t_aug_per_sample_ms = t_base_per_sample_ms + t_retrieval_per_sample_ms
        pct = (t_retrieval_per_sample_ms / t_base_per_sample_ms) * 100 if t_base_per_sample_ms > 0 else 0
        print("[Delta Time] total_samples=%d" % total_samples)
        print("[Delta Time] t_base_per_sample = %.3f ms" % t_base_per_sample_ms)
        print("[Delta Time] t_retrieval_per_sample (delta) = %.3f ms" % t_retrieval_per_sample_ms)
        print("[Delta Time] t_aug_per_sample = %.3f ms" % t_aug_per_sample_ms)
        print("[Delta Time] retrieval_overhead = %.2f%%" % pct)

    return pd.DataFrame(all_results)


def main(args):
    Retrieval_config["top_k"] = args.top_k
    Retrieval_config["enhancement_weight"] = args.enhancement_weight
    Retrieval_config["similarity_threshold"] = args.similarity_threshold
    print("Retrieval_config: ", Retrieval_config)
    
    # model
    model = LitModel.load_from_checkpoint(args.checkpoint_path)
    eval_model = model.model
    eval_model.to(args.device)

    mol_tokenizer = torch.load(args.mol_model_locate)
    
    # eval dataset
    evalset = EvalDataset(args)
    eval_loader = evalset.get_testloader()

    # 1. faiss
    print("Initializing FaissRAG...")
    faiss_cfg = FaissIndexConfig(args)
    searcher = DualIndexRAG(
        dim_enzyme=args.dim_enzyme,
        dim_mol=args.dim_mol,
        cfg=faiss_cfg,
    )
    # 2. index: protein and mol with CLEAN embedding
    # protein
    assert args.protein_target_embedding_file_path is not None, "protein_target_embedding_file_path is required"
    target_emb_dict = torch.load(args.protein_target_embedding_file_path, map_location="cpu")
    if len(target_emb_dict) == 0:
        raise ValueError("Empty target embedding dictionary")
    # build index
    enz_embs = torch.stack(list(target_emb_dict.values()))
    enz_seqs = list(target_emb_dict.keys())
    searcher.build_enzyme_index(enz_embs = enz_embs, enz_keys = enz_seqs)

    # mol if use molecule need initial it or dont
    if args.use_molecule_retrieval and args.mol_target_embedding_file_path is not None:
        assert args.mol_target_embedding_file_path is not None, "mol_target_embedding_file_path is required"
        target_emb_dict = torch.load(args.mol_target_embedding_file_path, map_location="cpu")
        if len(target_emb_dict) == 0:
            raise ValueError("Empty mol target embedding dictionary")
        mol_embs = torch.stack(list(target_emb_dict.values()))
        mol_seqs = list(target_emb_dict.keys())
        searcher.build_molecule_index(mol_embs = mol_embs, mol_keys = mol_seqs)
    print("Searcher initialized.")

    # 3. protein embedding dict: convert the searched seqs to embedding cuase we dont load pre-trained model
    # this embedding is the actual inference model used!   
    # eval_loader already contains the seqs/ mol/ embedding and query embedding(CLEAN) for the current query
    print("Creating sequence-to-path mapping dictionary...")
    target_embedding_df = get_embedding_df(args.protein_train_data_path)
    protein_seq_to_path_dict = pd.Series(
        target_embedding_df['Protein_Path'].values,
        index=target_embedding_df['Protein']
    ).to_dict()
    # ! 实际上并没有保存mol的路径而是直接全量加载了小分子的输入嵌入: mol_tokenizer
    mol_seq_to_path_dict=None
    if args.use_molecule_retrieval and args.mol_train_data_path is not None:
        target_embedding_df = get_embedding_df(args.mol_train_data_path)
        mol_seq_to_path_dict = pd.Series(
            target_embedding_df['Mol_Path'].values,
            index=target_embedding_df['Mol']
        ).to_dict()
    print("Mapping dictionary created.")

    # execute the inference, and pass the initialized objects
    # infer(args, model, eval_loader, mol_tokenizer, searcher, protein_seq_to_path_dict, mol_seq_to_path_dict=None)
    results = infer(args, eval_model, eval_loader, mol_tokenizer, searcher, protein_seq_to_path_dict=protein_seq_to_path_dict)
    # save results
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    results.to_csv(os.path.join(output_dir, args.output_file), index=False)
    print(f"Inference complete. Results saved to {os.path.join(output_dir, args.output_file)}")



if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_name", type=str, default="esp")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--protein_col_name", type=str, default="Protein")
    parser.add_argument("--protein_embed_col", type=str, default="Protein_Path")
    parser.add_argument("--mol_col_name", type=str, default="SMILES")
    parser.add_argument("--label_col_name", type=str, default="Y")
    parser.add_argument("--prot_model_locate", type=str, default=None)
    parser.add_argument("--dim_enzyme", type=int, default=128)
    parser.add_argument("--dim_mol", type=int, default=1536)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mol_train_data_path", type=str, default=None)
    parser.add_argument("--mol_target_embedding_file_path", type=str, default=None)
    
    # config
    parser.add_argument("--protein_target_embedding_file_path", type=str, default="./clean_embedding/clean_esp/esp_train_embedding.pt") # 检索库的embedding
    parser.add_argument("--mol_model_locate", type=str, default="./mol_repr/unimol_esp/esp_mol_repr.pt")
    parser.add_argument("--clean_embedding_path", type=str, default="./clean_embedding/clean_esp/esp_test_embedding.pt") # 查询库的embedding
    parser.add_argument("--checkpoint_path", type=str, default="./ckpt/esm2_650m_unimol-esp_unimol_esm_IFL/ckpt_lr-3e-05_patience-50.ckpt")
    parser.add_argument("--eval_file_path", type=str, default="./datasets/esp/test.csv") # 查询库的csv
    parser.add_argument("--protein_train_data_path", type=str, default="./datasets/esp/train.csv") # 检索库
    parser.add_argument("--output_dir", type=str, default="esp_res")
    parser.add_argument("--output_file", type=str, default="results.csv")
    
    
    # retrieval
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--enhancement_weight", type=float, default=0.2)
    parser.add_argument("--similarity_threshold", type=float, default=0.3)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--use_molecule_retrieval", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.5)
    # faiss
    parser.add_argument("--faiss_gpu", action="store_true") 
    parser.add_argument("--use_ivf", action="store_true") 
    parser.add_argument("--nlist", type=int, default=20)
    parser.add_argument("--nprobe", type=int, default=5)
    parser.add_argument("--m_pq", type=int, default=16)
    parser.add_argument("--index_method", type=str, default="flat_ip")
    parser.add_argument("--clamp_logit", type=float, default=8.0)
    

    args = parser.parse_args()
    main(args)