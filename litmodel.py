import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC, MatthewsCorrCoef, AveragePrecision
from collections import OrderedDict
from model import PMModel
from utils.loss_fn import ThanFocalLoss

class LitModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = PMModel(args)

        if args.num_labels == 1:
            self.criterion = nn.BCEWithLogitsLoss()
            metrics_kwargs = {"task": "binary"}
            if args.unbalance:
                print("use unbalance and ThanFocalLoss")
                self.criterion = ThanFocalLoss(gamma=args.gamma, alpha=args.alpha, theta=args.theta, reduction='mean')
                
        self.test_predictions = []
                            
        # val
        self.valid_metrics = nn.ModuleDict(
            OrderedDict([
                ("val_auroc", AUROC(**metrics_kwargs)),
                ("val_acc", Accuracy(**metrics_kwargs)),
                ("val_f1", F1Score(**metrics_kwargs)),
                ("val_precision", Precision(**metrics_kwargs)),
                ("val_recall", Recall(**metrics_kwargs)),
                ("val_mcc", MatthewsCorrCoef(**metrics_kwargs)),
                ("val_auprc", AveragePrecision(**metrics_kwargs))
            ])
        )
        self.test_metrics = nn.ModuleDict(
            OrderedDict([
                ("test_auroc", AUROC(**metrics_kwargs)),
                ("test_acc", Accuracy(**metrics_kwargs)),
                ("test_f1", F1Score(**metrics_kwargs)),
                ("test_precision", Precision(**metrics_kwargs)),
                ("test_recall", Recall(**metrics_kwargs)),
                ("test_mcc", MatthewsCorrCoef(**metrics_kwargs)),
                ("test_auprc",  AveragePrecision(**metrics_kwargs))
            ])
        )
        self.save_hyperparameters()
    
    
    def training_step(self, batch, batch_idx):
        
        labels = batch["labels"]
        protein_embeddings = batch["protein_embeddings"]
        protein_embedding_mask = batch["protein_embedding_mask"]
        mol_embeddings = batch["mol_embeddings"]
        mol_embedding_mask = batch["mol_embedding_mask"]
        outs = self.model(
            protein_embeddings=protein_embeddings,
            protein_embedding_mask=protein_embedding_mask,
            mol_embeddings=mol_embeddings,
            mol_embedding_mask=mol_embedding_mask,
        )
        if self.args.num_labels > 1:
            loss = self.criterion(outs, labels)
        else:
            loss = self.criterion(outs, labels.float())
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        protein_embeddings = batch["protein_embeddings"]
        protein_embedding_mask = batch["protein_embedding_mask"]
        mol_embeddings = batch["mol_embeddings"]
        mol_embedding_mask = batch["mol_embedding_mask"]
        outs = self.model(
            protein_embeddings=protein_embeddings,
            protein_embedding_mask=protein_embedding_mask,
            mol_embeddings=mol_embeddings,
            mol_embedding_mask=mol_embedding_mask,
        )
        probs = self._get_probabilities(outs)
        for name, metric in self.valid_metrics.items():
            metric(probs, labels)
            self.log(name, metric, on_epoch=True, prog_bar=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        labels = batch["labels"]
        protein_embeddings = batch["protein_embeddings"]
        protein_embedding_mask = batch["protein_embedding_mask"]
        mol_embeddings = batch["mol_embeddings"]
        mol_embedding_mask = batch["mol_embedding_mask"]
        
        outs = self.model(
            protein_embeddings=protein_embeddings,
            protein_embedding_mask=protein_embedding_mask,
            mol_embeddings=mol_embeddings,
            mol_embedding_mask=mol_embedding_mask,
        )
        
        probs = self._get_probabilities(outs)
        for name, metric in self.test_metrics.items():
            metric(probs, labels)
            self.log(name, metric, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # add prob
        self.test_predictions.extend(probs.cpu().tolist())
    
    def _get_probabilities(self, outputs):
        """根据任务类型获取概率分布"""
        if self.args.num_labels > 1:
            return F.softmax(outputs, dim=-1)
        else:
            return torch.sigmoid(outputs)
    
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.args.lr,
        )
        return optimizer
