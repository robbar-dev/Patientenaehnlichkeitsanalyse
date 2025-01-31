import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.base_cnn import BaseCNN
from model.mean_pooling import MeanPoolingAggregator
from training.data_loader import SinglePatientDataset
import matplotlib.pyplot as plt
import numpy as np

def combo_to_label(combo_str):
    """
    Ordnet den 3 relevanten Kombinationen => Label {0,1,2}
    '1-0-0' => label=0
    '0-1-0' => label=1
    '0-0-1' => label=2
    """
    if combo_str == "1-0-0":
        return 0
    elif combo_str == "0-1-0":
        return 1
    elif combo_str == "0-0-1":
        return 2
    else:
        # Falls wir nicht in den 3-Klassen-Fall sind => None
        return None

class ClassificationTrainer(nn.Module):
    """
    Einfache 3-Klassen-Klassifikation: combos { '1-0-0','0-1-0','0-0-1' } => label in {0,1,2}.
    - CNN + Aggregator => (1,512)
    - linear => (1,3)
    - CrossEntropy
    """

    def __init__(
        self,
        df,
        data_root,
        device='cuda',
        lr=1e-4,
        roi_size=(96,96,3),
        overlap=(10,10,1),
        skip_slices=True,
        skip_factor=2,
        filter_empty_patches=False,
        min_nonzero_fraction=0.01,
        filter_uniform_patches=False,
        min_std_threshold=0.01,
        do_patch_minmax=False,
        do_augmentation_train=True
    ):
        super().__init__()
        self.df = df.copy()
        self.data_root = data_root
        self.device = device
        self.roi_size = roi_size
        self.overlap = overlap

        self.skip_slices = skip_slices
        self.skip_factor = skip_factor
        self.filter_empty_patches = filter_empty_patches
        self.min_nonzero_fraction = min_nonzero_fraction
        self.filter_uniform_patches = filter_uniform_patches
        self.min_std_threshold = min_std_threshold
        self.do_patch_minmax = do_patch_minmax
        self.do_augmentation_train = do_augmentation_train

        # Filter df => nur combos=1-0-0,0-1-0,0-0-1
        valid_combos = {"1-0-0","0-1-0","0-0-1"}
        self.df = self.df[self.df["combination"].isin(valid_combos)].copy()
        # Mapping => label
        self.df["cls_label"] = self.df["combination"].apply(combo_to_label)
        # Filter rows where label=None
        self.df = self.df.dropna(subset=["cls_label"])
        self.df["cls_label"] = self.df["cls_label"].astype(int)

        # (A) CNN
        self.base_cnn = BaseCNN(model_name='resnet18', pretrained=False).to(device)

        # (B) Aggregator => MeanPooling
        self.aggregator = MeanPoolingAggregator().to(device)

        # (C) linear => 3-Klassen
        self.classifier = nn.Linear(512, 3).to(device)

        # Loss
        self.ce_loss_fn = nn.CrossEntropyLoss()

        # Optimizer
        params = list(self.base_cnn.parameters()) + list(self.aggregator.parameters()) + list(self.classifier.parameters())
        self.optimizer = optim.Adam(params, lr=lr)

        self.scheduler = None

        # Tracking
        self.epoch_losses = []
        self.epoch_train_acc = []  # train-acc each epoch
        self.epoch_val_acc = []    # val-acc every 5 epoch or so

        logging.info(f"[ClassificationTrainer] 3-Klassen Setup => combos in {valid_combos}")

    # --------------------------------------------------------------------------
    # forward => train
    # --------------------------------------------------------------------------
    def _forward_patient(self, pid, study_yr):
        """
        Holt Patches => CNN => aggregator => => (1,512) => classifier => (1,3) => logits
        => returns logits shape(1,3)
        """
        ds = SinglePatientDataset(
            data_root=self.data_root,
            pid=pid,
            study_yr=study_yr,
            roi_size=self.roi_size,
            overlap=self.overlap,
            skip_slices=self.skip_slices,
            skip_factor=self.skip_factor,
            filter_empty_patches=self.filter_empty_patches,
            min_nonzero_fraction=self.min_nonzero_fraction,
            filter_uniform_patches=self.filter_uniform_patches,
            min_std_threshold=self.min_std_threshold,
            do_patch_minmax=self.do_patch_minmax,
            do_augmentation=self.do_augmentation_train
        )
        loader = DataLoader(ds, batch_size=16, shuffle=False)

        self.base_cnn.train()
        self.aggregator.train()
        self.classifier.train()

        patch_embs = []
        for patch_t in loader:
            patch_t = patch_t.to(self.device)
            emb = self.base_cnn(patch_t)  # => (B,512)
            patch_embs.append(emb)

        if len(patch_embs)==0:
            # leeres Volume => dummy
            dummy = torch.zeros((1,512), device=self.device, requires_grad=True)
            logits = self.classifier(dummy) # => shape(1,3)
            return logits

        patch_embs = torch.cat(patch_embs, dim=0)  # => (N,512)
        patient_emb = self.aggregator(patch_embs)   # => (1,512)
        logits = self.classifier(patient_emb)       # => (1,3)
        return logits

    # --------------------------------------------------------------------------
    # forward => evaluation
    # --------------------------------------------------------------------------
    def compute_patient_logits_eval(self, pid, study_yr):
        """
        -> eval-mode => no augmentation
        """
        ds = SinglePatientDataset(
            data_root=self.data_root,
            pid=pid,
            study_yr=study_yr,
            roi_size=self.roi_size,
            overlap=self.overlap,
            skip_slices=self.skip_slices,
            skip_factor=self.skip_factor,
            filter_empty_patches=self.filter_empty_patches,
            min_nonzero_fraction=self.min_nonzero_fraction,
            filter_uniform_patches=self.filter_uniform_patches,
            min_std_threshold=self.min_std_threshold,
            do_patch_minmax=self.do_patch_minmax,
            do_augmentation=False
        )
        loader = DataLoader(ds, batch_size=16, shuffle=False)

        self.base_cnn.eval()
        self.aggregator.eval()
        self.classifier.eval()

        patch_embs = []
        with torch.no_grad():
            for patch_t in loader:
                patch_t = patch_t.to(self.device)
                emb = self.base_cnn(patch_t)
                patch_embs.append(emb)

        if len(patch_embs)==0:
            dummy = torch.zeros((1,512), device=self.device)
            logits = self.classifier(dummy)
            return logits  # => shape(1,3)

        patch_embs = torch.cat(patch_embs, dim=0)
        patient_emb = self.aggregator(patch_embs)
        logits = self.classifier(patient_emb)
        return logits  # => shape(1,3)

    # --------------------------------------------------------------------------
    # train_loop
    # --------------------------------------------------------------------------
    def train_loop(self, num_epochs=10, df_val=None):
        """
        train df => shuffle => batch=1 patient => forward => crossent => step
        => at end epoch => compute train acc
        => alle 5 epoche => compute val acc (falls df_val != None)
        """
        df_patients = self.df.sample(frac=1.0).reset_index(drop=True)

        for epoch in range(1, num_epochs+1):
            total_loss = 0.0
            steps = 0
            correct = 0
            total = 0

            # --- TRAIN: go over train df ---
            for i, row in df_patients.iterrows():
                pid = row['pid']
                sy = row['study_yr']
                label = int(row['cls_label'])  # 0..2

                logits = self._forward_patient(pid, sy)  # => shape(1,3)
                # crossent
                target = torch.tensor([label], dtype=torch.long, device=self.device)  # shape(1,)
                loss = self.ce_loss_fn(logits, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                steps += 1

                # Quick train-acc
                _, pred_cls = torch.max(logits, dim=1)   # => (1,)
                if pred_cls.item() == label:
                    correct += 1
                total += 1

            avg_loss = total_loss/steps if steps>0 else 0
            train_acc = correct/total if total>0 else 0.0

            self.epoch_losses.append(avg_loss)
            self.epoch_train_acc.append(train_acc)

            msg = f"[Epoch {epoch}] CE-Loss={avg_loss:.4f}, train_acc={train_acc*100:.2f}%"

            # --- optional: VALIDATION ACC alle 5 Epoche ---
            if df_val is not None and (epoch % 5 == 0):
                val_loss, val_acc = self.eval_on_df(df_val)
                self.epoch_val_acc.append(val_acc)
                msg += f", val_acc={val_acc*100:.2f}%"

            logging.info(msg)

    # --------------------------------------------------------------------------
    # eval_on_df => returns loss,acc
    # --------------------------------------------------------------------------
    def eval_on_df(self, df_eval):
        """
        Einfache Evaluierung => ACC
        """
        correct = 0
        total = 0
        total_loss = 0
        steps = 0

        self.base_cnn.eval()
        self.aggregator.eval()
        self.classifier.eval()

        with torch.no_grad():
            for i, row in df_eval.iterrows():
                pid = row['pid']
                sy = row['study_yr']
                label = int(row['cls_label'])

                logits = self.compute_patient_logits_eval(pid, sy)
                target = torch.tensor([label], dtype=torch.long, device=self.device)
                loss = self.ce_loss_fn(logits, target)
                total_loss += loss.item()
                steps+=1

                _, pred_cls = torch.max(logits, dim=1)
                if pred_cls.item() == label:
                    correct+=1
                total+=1

        avg_loss = total_loss/steps if steps>0 else 0
        acc = correct/total if total>0 else 0
        return avg_loss, acc

    # --------------------------------------------------------------------------
    # plot_loss_acc
    # --------------------------------------------------------------------------
    def plot_loss_acc(self):
        """
        Zeichnet:
         1) den Verlauf des CrossEntropy-Loss => cls_loss.png
         2) den Verlauf der TRAIN-Accuracy => cls_train_acc.png
         3) den Verlauf der VAL-Accuracy => cls_val_acc.png (nur alle 5 Epochen => also length<=> E/5)
        """
        if not self.epoch_losses:
            logging.info("Keine epoch_losses => skip plot_loss_acc.")
            return

        import os
        import matplotlib.pyplot as plt
        import numpy as np

        if not os.path.exists("plots_cls"):
            os.makedirs("plots_cls")

        # --- 1) Loss Plot ---
        epochs_range = range(1, len(self.epoch_losses)+1)
        plt.figure(figsize=(6,5))
        plt.plot(epochs_range, self.epoch_losses, 'r-o', label='CE-Loss')
        plt.title("Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        out_loss = os.path.join("plots_cls", "cls_loss.png")
        plt.savefig(out_loss, dpi=150)
        plt.close()
        logging.info(f"Saved Loss Plot => {out_loss}")

        # --- 2) TRAIN Accuracy Plot ---
        if len(self.epoch_train_acc)>0:
            plt.figure(figsize=(6,5))
            plt.plot(epochs_range, self.epoch_train_acc, 'b-s', label='Train Accuracy')
            plt.title("Train Accuracy vs Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.ylim(0,1)
            plt.legend()
            out_train_acc = os.path.join("plots_cls", "cls_train_acc.png")
            plt.savefig(out_train_acc, dpi=150)
            plt.close()
            logging.info(f"Saved Train Accuracy Plot => {out_train_acc}")

        # --- 3) VAL Accuracy Plot ---
        val_epochs = [5 * i for i in range(1, len(self.epoch_val_acc)+1)]
        if len(self.epoch_val_acc)>0:
            plt.figure(figsize=(6,5))
            plt.plot(val_epochs, self.epoch_val_acc, 'g-o', label='Val Accuracy (every 5 ep)')
            plt.title("Validation Accuracy vs Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.ylim(0,1)
            plt.legend()
            out_val_acc = os.path.join("plots_cls", "cls_val_acc.png")
            plt.savefig(out_val_acc, dpi=150)
            plt.close()
            logging.info(f"Saved Val Accuracy Plot => {out_val_acc}")
