import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import csv
from torch.utils.data import DataLoader

from model.base_cnn import BaseCNN
from model.mil_aggregator import AttentionMILAggregator
from training.data_loader import SinglePatientDataset

import matplotlib.pyplot as plt

def combo_to_label(combo_str):
    if combo_str == "1-0-0":
        return 0
    elif combo_str == "0-1-0":
        return 1
    elif combo_str == "0-0-1":
        return 2
    return None

class ClassificationTrainer(nn.Module):
    def __init__(
        self,
        df,
        data_root,
        device='cuda',
        lr=1e-4,
        roi_size=(96,96,3),
        overlap=(10,10,1),
        skip_factor=2,
        do_augmentation_train=False,
        csv_path="classification_results.csv"
    ):
        super().__init__()
        self.df = df.copy()
        self.data_root = data_root
        self.device = device
        self.lr = lr
        self.roi_size = roi_size
        self.overlap = overlap
        self.skip_factor = skip_factor
        self.do_augmentation_train = do_augmentation_train
        self.csv_path = csv_path

        self.base_cnn = BaseCNN(
            model_name='resnet18',
            pretrained=True, 
            freeze_blocks=[0,1]
        ).to(device)

        self.aggregator = AttentionMILAggregator(
            in_dim=512,
            hidden_dim=128,
            dropout=0.2
        ).to(device)

        # Klassifikationskopf
        self.classifier = nn.Linear(512, 3).to(device)

        # CrossEntropy
        self.ce_loss_fn = nn.CrossEntropyLoss()

        # Optimizer 
        params = list(self.base_cnn.parameters()) \
               + list(self.aggregator.parameters()) \
               + list(self.classifier.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr)

        self.scheduler = None

        self.epoch_losses = []
        self.epoch_train_acc = []
        self.epoch_val_acc = []

        logging.info("[ClassificationTrainer] Using AttentionMIL for 3-class classification.")
        logging.info(f"lr={lr}, roi_size={roi_size}, overlap={overlap}, skip_factor={skip_factor}, "
                     f"do_augmentation_train={do_augmentation_train}")
        
        self._prepare_csv()

    def _prepare_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy"])
            logging.info(f"Classification - CSV-Datei erstellt: {self.csv_path}")

    def _write_to_csv(self, epoch, train_loss, train_acc, val_loss=None, val_acc=None):
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            row = [epoch, train_loss, train_acc]
            if val_loss is not None and val_acc is not None:
                row.extend([val_loss, val_acc])
            else:
                row.extend(["-", "-"])
            writer.writerow(row)

    def _forward_patient(self, pid, study_yr):
        ds = SinglePatientDataset(
            data_root=self.data_root,
            pid=pid,
            study_yr=study_yr,
            roi_size=self.roi_size,
            overlap=self.overlap,
            skip_factor=self.skip_factor,
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
            dummy = torch.zeros((1,512), device=self.device, requires_grad=True)
            logits = self.classifier(dummy)
            return logits  # shape(1,3)

        patch_embs = torch.cat(patch_embs, dim=0)  # => (N,512)
        patient_emb = self.aggregator(patch_embs)  # => (1,512)
        logits = self.classifier(patient_emb)       # => (1,3)
        return logits

    def compute_patient_logits_eval(self, pid, study_yr):
        ds = SinglePatientDataset(
            data_root=self.data_root,
            pid=pid,
            study_yr=study_yr,
            roi_size=self.roi_size,
            overlap=self.overlap,
            skip_factor=self.skip_factor,
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
            return self.classifier(dummy)  # => shape(1,3)

        patch_embs = torch.cat(patch_embs, dim=0)
        patient_emb = self.aggregator(patch_embs)
        logits = self.classifier(patient_emb)
        return logits

    def train_loop(self, num_epochs=10, df_val=None):
        df_patients = self.df.sample(frac=1.0).reset_index(drop=True)

        for epoch in range(1, num_epochs+1):
            total_loss = 0.0
            steps = 0
            correct = 0
            total = 0

            for i, row in df_patients.iterrows():
                pid = row['pid']
                sy  = row['study_yr']
                combo_str = row['combination']
                label_int = combo_to_label(combo_str)  # => 0..2

                if label_int is None:
                    continue

                logits = self._forward_patient(pid, sy)
                target = torch.tensor([label_int], dtype=torch.long, device=self.device)

                loss = self.ce_loss_fn(logits, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                steps += 1

                _, pred_cls = torch.max(logits, dim=1)
                if pred_cls.item() == label_int:
                    correct += 1
                total += 1

            avg_loss = total_loss/steps if steps>0 else 0
            train_acc = correct/total if total>0 else 0
            self.epoch_losses.append(avg_loss)
            self.epoch_train_acc.append(train_acc)

            msg = f"[Epoch {epoch}] CE-Loss={avg_loss:.4f}, train_acc={train_acc*100:.2f}%"

            if df_val is not None:
                val_loss, val_acc = self.eval_on_df(df_val)
                self.epoch_val_acc.append(val_acc)
                msg += f", val_acc={val_acc*100:.2f}%"

            logging.info(msg)

            self._write_to_csv(epoch, avg_loss, train_acc, val_loss, val_acc)

            if self.scheduler is not None:
                self.scheduler.step()

    def eval_on_df(self, df_eval):
        correct = 0
        total = 0
        total_loss = 0.0
        steps = 0

        self.base_cnn.eval()
        self.aggregator.eval()
        self.classifier.eval()

        with torch.no_grad():
            for i, row in df_eval.iterrows():
                pid = row['pid']
                sy  = row['study_yr']
                combo_str = row['combination']
                label_int = combo_to_label(combo_str)
                if label_int is None:
                    continue

                logits = self.compute_patient_logits_eval(pid, sy)
                target = torch.tensor([label_int], dtype=torch.long, device=self.device)

                loss = self.ce_loss_fn(logits, target)
                total_loss += loss.item()
                steps += 1

                _, pred_cls = torch.max(logits, dim=1)
                if pred_cls.item() == label_int:
                    correct += 1
                total += 1

        avg_loss = total_loss/steps if steps>0 else 0
        acc = correct/total if total>0 else 0
        return avg_loss, acc

    def plot_loss_acc(self, out_dir="plots_cls"):
        import os
        import matplotlib.pyplot as plt

        if not self.epoch_losses:
            logging.info("Keine epoch_losses => skip plot.")
            return

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        epochs_range = range(1, len(self.epoch_losses)+1)

        # CE-Loss
        plt.figure(figsize=(6,5))
        plt.plot(epochs_range, self.epoch_losses, 'r-o', label='CE-Loss')
        plt.title("Train Loss (CE)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        out_loss = os.path.join(out_dir, "cls_loss.png")
        plt.savefig(out_loss, dpi=150)
        plt.close()
        logging.info(f"Saved Loss Plot => {out_loss}")

        # Train-Accuracy
        plt.figure(figsize=(6,5))
        plt.plot(epochs_range, self.epoch_train_acc, 'b-s', label='Train Accuracy')
        plt.title("Train Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0,1)
        plt.legend()
        out_train_acc = os.path.join(out_dir, "cls_train_acc.png")
        plt.savefig(out_train_acc, dpi=150)
        plt.close()
        logging.info(f"Saved Train Accuracy Plot => {out_train_acc}")

        # Val-Accuracy
        plt.figure(figsize=(6,5))
        val_epochs = range(1, len(self.epoch_val_acc)+1)
        plt.plot(val_epochs, self.epoch_val_acc, 'g-o', label='Val Accuracy')
        plt.title("Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0,1)
        plt.legend()
        out_val_acc = os.path.join(out_dir, "cls_val_acc.png")
        plt.savefig(out_val_acc, dpi=150)
        plt.close()
        logging.info(f"Saved Val Accuracy Plot => {out_val_acc}")
