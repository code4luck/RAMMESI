import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class MetricsToFileCallback(Callback):
    def __init__(self, folder):
        self.folder = folder
        self.file_path = os.path.join(self.folder, "results.csv")
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.fieldnames = ["acc", "f1", "precision", "recall", "auroc", "mcc", "auprc"]

    def on_test_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            metrics = {
                "acc": trainer.logged_metrics.get("test_acc", None),
                "f1": trainer.logged_metrics.get("test_f1", None),
                "precision": trainer.logged_metrics.get("test_precision", None),
                "recall": trainer.logged_metrics.get("test_recall", None),
                "auroc": trainer.logged_metrics.get("test_auroc", None),
                "mcc": trainer.logged_metrics.get("test_mcc", None),
                "auprc": trainer.logged_metrics.get("test_auprc", None),
                # "epoch": trainer.current_epoch,
            }
            # None replace with pd.NA, convert to float
            # metrics = {
            #     k: (pd.NA if v is None else float(v)) if k != "epoch" else v
            #     for k, v in metrics.items()
            # }
            metrics = {k: (pd.NA if v is None else float(v)) for k, v in metrics.items()}
            df_new = pd.DataFrame([metrics], columns=self.fieldnames)
            
            if os.path.exists(self.file_path):
                df_new.to_csv(self.file_path, mode="a", header=False, index=False)
            else:
                df_new.to_csv(self.file_path, mode="w", header=True, index=False)
