import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import datetime
import csv

from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 1) CNN-Backbone
from model.base_cnn import BaseCNN
# 2) Aggregatoren
from model.mil_aggregator import AttentionMILAggregator
from model.max_pooling import MaxPoolingAggregator
from model.mean_pooling import MeanPoolingAggregator
# 3) TripletSampler
from training.triplet_sampler import TripletSampler
# 4) HardNegativeTripletSampler
from training.triplet_sampler_hard_negativ import HardNegativeTripletSampler
# 5) SinglePatientDataset
from training.data_loader import SinglePatientDataset
# 6) Evaluate-Funktionen (IR-Metriken)
from evaluation.metrics import evaluate_model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Für ACC, AUC-Berechnung
from sklearn.metrics import accuracy_score, roc_auc_score

# ------------------------------------------------
# (1) Funktion zum Parsen: "1-0-0" => 1 (krank), "0-0-1" => 0 (gesund)
# ------------------------------------------------
def parse_combo_str_to_vec(combo_str):
    parts = [int(x) for x in combo_str.split('-')]
    if parts == [1,0,0]:
        return 1
    else:
        return 0


class TripletTrainer(nn.Module):
    def __init__(
        self,
        aggregator_name,
        df,
        data_root,
        device='cuda',
        lr=1e-3,
        margin=1.0,
        roi_size=(96, 96, 3),
        overlap=(10,10,1),
        pretrained=False,
        attention_hidden_dim=128,
        dropout=0.2,
        weight_decay=1e-5,
        freeze_blocks=None,
        skip_slices=True,
        skip_factor=2,
        filter_empty_patches=False,
        min_nonzero_fraction=0.01,
        filter_uniform_patches=True,
        min_std_threshold=0.01,
        do_patch_minmax=False,
        lambda_bce=1.0,
        do_augmentation_train=True
    ):
        super().__init__()
        self.df = df
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

        self.lambda_bce = lambda_bce
        self.do_augmentation_train = do_augmentation_train

        # (A) CNN-Backbone
        self.base_cnn = BaseCNN(
            model_name='resnet18',
            pretrained=pretrained,
            freeze_blocks=freeze_blocks
        ).to(device)

        # (B) Aggregator
        if aggregator_name == "mil":
            self.mil_agg = AttentionMILAggregator(
                in_dim=512,
                hidden_dim=attention_hidden_dim,
                dropout=dropout
            ).to(device)
        elif aggregator_name == "max":
            self.mil_agg = MaxPoolingAggregator().to(device)
        elif aggregator_name == "mean":
            self.mil_agg = MeanPoolingAggregator().to(device)
        else:
            raise ValueError(f"Unbekannter Aggregator: {aggregator_name}")

        # (C) TripletLoss
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

        # (D) Binary-Kopf + BCE-Loss => 1 Output (gesund=0 vs. krank=1)
        self.classifier = nn.Linear(512, 1).to(device)
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

        # (E) Optimizer
        params = list(self.base_cnn.parameters()) \
               + list(self.mil_agg.parameters()) \
               + list(self.classifier.parameters())
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

        # (F) Scheduler (optional)
        self.scheduler = None

        # Tracking
        self.best_val_map = 0.0
        self.best_val_epoch = -1

        # Verlaufslisten (Loss)
        self.epoch_losses = []         
        self.epoch_triplet_losses = []
        self.epoch_bce_losses = []

        # IR-Metriken
        self.metric_history = {
            "precision": [],
            "recall": [],
            "mAP": []
        }

        # NEU: Dictionary für binäre Klassifikation
        self.binclass_history = {
            "acc": [],
            "auc": []
        }

        logging.info(
            f"[TripletTrainer] aggregator={aggregator_name}, lr={lr}, margin={margin}, "
            f"dropout={dropout}, weight_decay={weight_decay}, freeze_blocks={freeze_blocks}, "
            f"skip_slices={skip_slices}, skip_factor={skip_factor}, "
            f"filter_empty_patches={filter_empty_patches}, min_nonzero_frac={min_nonzero_fraction}, "
            f"filter_uniform_patches={filter_uniform_patches}, min_std_threshold={min_std_threshold}, "
            f"do_patch_minmax={do_patch_minmax}, lambda_bce={lambda_bce}, "
            f"do_augmentation_train={do_augmentation_train}"
        )

    # ---------------------------
    # _forward_patient => (emb, logits)
    # ---------------------------
    def _forward_patient(self, pid, study_yr):
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

        loader = DataLoader(ds, batch_size=32, shuffle=False)

        self.base_cnn.train()
        self.mil_agg.train()
        self.classifier.train()

        patch_embs = []
        for patch_t in loader:
            patch_t = patch_t.to(self.device)
            emb = self.base_cnn(patch_t)
            patch_embs.append(emb)

        if len(patch_embs) == 0:
            dummy = torch.zeros((1,512), device=self.device, requires_grad=True)
            dummy_logits = self.classifier(dummy)
            return dummy, dummy_logits

        patch_embs = torch.cat(patch_embs, dim=0)
        patient_emb = self.mil_agg(patch_embs)
        logits = self.classifier(patient_emb)
        return patient_emb, logits

    def compute_patient_embedding(self, pid, study_yr):
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
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        self.base_cnn.eval()
        self.mil_agg.eval()
        self.classifier.eval()

        all_embs = []
        with torch.no_grad():
            for patch_t in loader:
                patch_t = patch_t.to(self.device)
                emb = self.base_cnn(patch_t)
                all_embs.append(emb)

        if len(all_embs)==0:
            return torch.zeros((1,512), device=self.device)

        patch_embs = torch.cat(all_embs, dim=0)
        patient_emb = self.mil_agg(patch_embs)
        return patient_emb

    # ---------------------------
    # train_one_epoch
    # ---------------------------
    def train_one_epoch(self, sampler):
        total_loss = 0.0
        total_trip = 0.0
        total_bce  = 0.0
        steps = 0

        for step, (anchor_info, pos_info, neg_info) in enumerate(sampler):
            a_pid, a_sy = anchor_info['pid'], anchor_info['study_yr']
            p_pid, p_sy = pos_info['pid'], pos_info['study_yr']
            n_pid, n_sy = neg_info['pid'], neg_info['study_yr']

            a_label = anchor_info['label']
            p_label = pos_info['label']
            n_label = neg_info['label']

            a_emb, a_logits = self._forward_patient(a_pid, a_sy)
            p_emb, p_logits = self._forward_patient(p_pid, p_sy)
            n_emb, n_logits = self._forward_patient(n_pid, n_sy)

            trip_loss = self.triplet_loss_fn(a_emb, p_emb, n_emb)

            a_label_t = torch.tensor([[a_label]], dtype=torch.float32, device=self.device)
            p_label_t = torch.tensor([[p_label]], dtype=torch.float32, device=self.device)
            n_label_t = torch.tensor([[n_label]], dtype=torch.float32, device=self.device)

            bce_a = self.bce_loss_fn(a_logits, a_label_t)
            bce_p = self.bce_loss_fn(p_logits, p_label_t)
            bce_n = self.bce_loss_fn(n_logits, n_label_t)
            bce_loss = (bce_a + bce_p + bce_n) / 3.0

            loss = trip_loss + self.lambda_bce * bce_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_trip += trip_loss.item()
            total_bce  += bce_loss.item()
            steps += 1

            if step % 50 == 0:
                logging.info(f"[Step {step}] TotalLoss={loss.item():.4f}, "
                             f"Trip={trip_loss.item():.4f}, BCE={bce_loss.item():.4f}")

        if steps>0:
            avg_loss = total_loss/steps
            avg_trip = total_trip/steps
            avg_bce  = total_bce/steps
        else:
            avg_loss=0; avg_trip=0; avg_bce=0

        logging.info(f"Epoch Loss={avg_loss:.4f}, Trip={avg_trip:.4f}, BCE={avg_bce:.4f}")
        self.epoch_losses.append(avg_loss)
        self.epoch_triplet_losses.append(avg_trip)
        self.epoch_bce_losses.append(avg_bce)

    # ---------------------------
    # train_one_epoch_internal (Hard Negatives)
    # ---------------------------
    def _train_one_epoch_internal(self, sampler):
        total_loss = 0.0
        total_trip = 0.0
        total_bce  = 0.0
        steps = 0

        for step, (anchor_info, pos_info, neg_info) in enumerate(sampler):
            a_pid, a_sy = anchor_info['pid'], anchor_info['study_yr']
            p_pid, p_sy = pos_info['pid'], pos_info['study_yr']
            n_pid, n_sy = neg_info['pid'], neg_info['study_yr']

            a_label = anchor_info['label']
            p_label = pos_info['label']
            n_label = neg_info['label']

            a_emb, a_logits = self._forward_patient(a_pid, a_sy)
            p_emb, p_logits = self._forward_patient(p_pid, p_sy)
            n_emb, n_logits = self._forward_patient(n_pid, n_sy)

            trip_loss = self.triplet_loss_fn(a_emb, p_emb, n_emb)

            a_label_t = torch.tensor([[a_label]], dtype=torch.float32, device=self.device)
            p_label_t = torch.tensor([[p_label]], dtype=torch.float32, device=self.device)
            n_label_t = torch.tensor([[n_label]], dtype=torch.float32, device=self.device)

            bce_a = self.bce_loss_fn(a_logits, a_label_t)
            bce_p = self.bce_loss_fn(p_logits, p_label_t)
            bce_n = self.bce_loss_fn(n_logits, n_label_t)
            bce_loss = (bce_a + bce_p + bce_n) / 3.0

            loss = trip_loss + self.lambda_bce*bce_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_trip += trip_loss.item()
            total_bce  += bce_loss.item()
            steps += 1

            if step % 250 == 0:
                logging.info(f"[Step {step}] TotalLoss={loss.item():.4f}, "
                             f"Trip={trip_loss.item():.4f}, BCE={bce_loss.item():.4f}")

        if steps>0:
            avg_loss = total_loss/steps
            avg_trip = total_trip/steps
            avg_bce  = total_bce/steps
        else:
            avg_loss=0; avg_trip=0; avg_bce=0

        logging.info(f"Epoch Loss={avg_loss:.4f}, Trip={avg_trip:.4f}, BCE={avg_bce:.4f}")
        self.epoch_losses.append(avg_loss)
        self.epoch_triplet_losses.append(avg_trip)
        self.epoch_bce_losses.append(avg_bce)

    # ---------------------------
    # train_loop
    # ---------------------------
    def train_loop(self, num_epochs=2, num_triplets=100):
        sampler = HardNegativeTripletSampler(
            df=self.df,
            trainer=self,
            num_triplets=num_triplets,
            device=self.device
        )

        self.train_one_epoch(sampler)

        for epoch in range(1, num_epochs+1):
            self.train_one_epoch(sampler)

            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                logging.info(f"Nach Epoche {epoch}, LR={current_lr}")

            sampler.reset_epoch()

    # ---------------------------
    # train_with_val
    # ---------------------------
    def train_with_val(
        self,
        epochs,
        num_triplets,
        val_csv,
        data_root_val,
        K=3,
        distance_metric='euclidean',
        visualize_every=5,
        visualize_method='tsne',
        output_dir='plots'
    ):
        self.best_val_map = 0.0
        self.best_val_epoch = -1

        # Verlaufslisten leeren
        self.epoch_losses = []
        self.epoch_triplet_losses = []
        self.epoch_bce_losses = []
        self.metric_history = {"precision": [], "recall": [], "mAP": []}
        self.binclass_history = {"acc": [], "auc": []}

        df_val = pd.read_csv(val_csv)

        for epoch in range(1, epochs+1):
            logging.info(f"=== EPOCH {epoch}/{epochs} ===")
            self.train_loop(num_epochs=1, num_triplets=num_triplets)

            # (1) IR-Metriken
            val_metrics = evaluate_model(
                trainer=self,
                data_csv=val_csv,
                data_root=data_root_val,
                K=K,
                distance_metric=distance_metric,
                device=self.device
            )
            precisionK = val_metrics["precision@K"]
            recallK    = val_metrics["recall@K"]
            map_val    = val_metrics["mAP"]

            logging.info(f"Val-Epoch={epoch}: precision={precisionK:.4f}, "
                         f"recall={recallK:.4f}, mAP={map_val:.4f}")
            self.metric_history["precision"].append(precisionK)
            self.metric_history["recall"].append(recallK)
            self.metric_history["mAP"].append(map_val)

            if map_val > self.best_val_map:
                self.best_val_map = map_val
                self.best_val_epoch = epoch
                logging.info(f"=> New Best mAP={map_val:.4f} @ epoch={epoch}")

            # (2) Binary classification => ACC, AUC
            acc_val, auc_val = self.evaluate_binary_classification(df_val, data_root_val)
            self.binclass_history["acc"].append(acc_val)
            self.binclass_history["auc"].append(auc_val)
            logging.info(f"Binary-Classification => ACC={acc_val:.4f}, AUC={auc_val:.4f}")

            # (3) Visualisierung
            if epoch % visualize_every == 0:
                self.visualize_embeddings(
                    df=df_val,
                    data_root=data_root_val,
                    method=visualize_method,
                    epoch=epoch,
                    output_dir=output_dir
                )

        self.plot_loss_components(output_dir=output_dir)
        self.plot_metric_curves(output_dir=output_dir)
        self.plot_acc_auc_curves(output_dir=output_dir)  # <--- Neu
        return (self.best_val_map, self.best_val_epoch)

    # ---------------------------
    # train_with_val_2stage
    # ---------------------------
    def train_with_val_2stage(
        self,
        epochs_stage1,
        epochs_stage2,
        num_triplets,
        val_csv,
        data_root_val,
        K=3,
        distance_metric='euclidean',
        visualize_every=5,
        visualize_method='tsne',
        output_dir='plots',
        epoch_csv_path=None
    ):
        if epoch_csv_path is not None:
            if not os.path.exists(epoch_csv_path):
                with open(epoch_csv_path, mode='w', newline='') as f:
                    writer = csv.writer(f, delimiter=';')
                    header = [
                        "Stage", "Epoch", "TotalLoss", "TripletLoss", "BCELoss",
                        "Precision@K", "Recall@K", "mAP",
                        "ACC", "AUC"
                    ]
                    writer.writerow(header)

        df_val = pd.read_csv(val_csv)

        # Reset
        self.best_val_map = 0.0
        self.best_val_epoch = -1
        self.epoch_losses = []
        self.epoch_triplet_losses = []
        self.epoch_bce_losses = []
        self.metric_history = {"precision": [], "recall": [], "mAP": []}
        self.binclass_history = {"acc": [], "auc": []}

        total_epochs = epochs_stage1 + epochs_stage2
        current_epoch = 0

        # (A) Stage1
        normal_sampler = TripletSampler(df=self.df, num_triplets=num_triplets,
                                        shuffle=True, top_k_negatives=3)

        for e in range(1, epochs_stage1+1):
            current_epoch += 1
            logging.info(f"=== STAGE1-Epoch {current_epoch}/{total_epochs} ===")
            self._train_one_epoch_internal(normal_sampler)
            normal_sampler.reset_epoch()

            self._evaluate_and_visualize(
                current_epoch=current_epoch,
                total_epochs=total_epochs,
                val_csv=val_csv,
                df_val=df_val,
                data_root_val=data_root_val,
                K=K,
                distance_metric=distance_metric,
                visualize_every=visualize_every,
                visualize_method=visualize_method,
                output_dir=output_dir,
                stage="Stage1",
                epoch_csv_path=epoch_csv_path
            )

        # (B) Stage2
        hard_sampler = None
        for e in range(1, epochs_stage2+1):
            current_epoch += 1
            logging.info(f"=== STAGE2-Epoch {current_epoch}/{total_epochs} ===")

            if (e % 5) == 1:
                hard_sampler = HardNegativeTripletSampler(
                    df=self.df,
                    trainer=self,
                    num_triplets=num_triplets,
                    device=self.device
                )
            self._train_one_epoch_internal(hard_sampler)

            self._evaluate_and_visualize(
                current_epoch=current_epoch,
                total_epochs=total_epochs,
                val_csv=val_csv,
                df_val=df_val,
                data_root_val=data_root_val,
                K=K,
                distance_metric=distance_metric,
                visualize_every=visualize_every,
                visualize_method=visualize_method,
                output_dir=output_dir,
                stage="Stage2",
                epoch_csv_path=epoch_csv_path
            )

        self.plot_loss_components(output_dir=output_dir)
        self.plot_metric_curves(output_dir=output_dir)
        self.plot_acc_auc_curves(output_dir=output_dir)
        return self.best_val_map, self.best_val_epoch

    # ---------------------------
    # evaluate_binary_classification => ACC, AUC
    # ---------------------------
    def evaluate_binary_classification(self, df, data_root):
        """
        1) Für jeden Patienten => Logit => Sigmoid => prob
        2) Sammle alle gt-Labels + predicted probs
        3) ACC, AUC
        """
        self.base_cnn.eval()
        self.mil_agg.eval()
        self.classifier.eval()

        y_true = []
        y_scores = []

        for idx, row in df.iterrows():
            pid = row['pid']
            study_yr = row['study_yr']
            combo_str = row['combination']
            label = parse_combo_str_to_vec(combo_str)  # 0 oder 1

            with torch.no_grad():
                emb = self.compute_patient_embedding(pid, study_yr)
                logits = self.classifier(emb)  # (1,1)
                prob = torch.sigmoid(logits).item()

            y_true.append(label)
            y_scores.append(prob)

        # Accuracy => Schwelle 0.5
        y_pred = [1 if s>=0.5 else 0 for s in y_scores]
        acc = accuracy_score(y_true, y_pred)

        # ROC-AUC => braucht mind. eine pos+neg Instanz
        auc_val = 0.0
        if len(set(y_true))>1:
            auc_val = roc_auc_score(y_true, y_scores)

        return acc, auc_val

    # ---------------------------
    # _evaluate_and_visualize
    # ---------------------------
    def _evaluate_and_visualize(
        self,
        current_epoch,
        total_epochs,
        val_csv,
        df_val,
        data_root_val,
        K=3,
        distance_metric='euclidean',
        visualize_every=5,
        visualize_method='tsne',
        output_dir='plots',
        stage="Stage?",
        epoch_csv_path=None
    ):
        # IR-Metriken
        val_metrics = evaluate_model(
            trainer=self,
            data_csv=val_csv,
            data_root=data_root_val,
            K=K,
            distance_metric=distance_metric,
            device=self.device
        )
        precisionK = val_metrics["precision@K"]
        recallK    = val_metrics["recall@K"]
        map_val    = val_metrics["mAP"]

        logging.info(f"Val-Epoch={current_epoch}: precision={precisionK:.4f}, "
                     f"recall={recallK:.4f}, mAP={map_val:.4f}")
        self.metric_history["precision"].append(precisionK)
        self.metric_history["recall"].append(recallK)
        self.metric_history["mAP"].append(map_val)

        if map_val>self.best_val_map:
            self.best_val_map = map_val
            self.best_val_epoch = current_epoch
            logging.info(f"=> New Best mAP={map_val:.4f} @ epoch={current_epoch}")

        # Binär-Klassifikation => ACC, AUC
        acc_val, auc_val = self.evaluate_binary_classification(df_val, data_root_val)
        self.binclass_history["acc"].append(acc_val)
        self.binclass_history["auc"].append(auc_val)
        logging.info(f"Binary => ACC={acc_val:.4f}, AUC={auc_val:.4f}")

        # Visualisierung
        if current_epoch % visualize_every == 0:
            self.visualize_embeddings(
                df=df_val,
                data_root=data_root_val,
                method=visualize_method,
                epoch=current_epoch,
                output_dir=output_dir
            )

        # CSV-Logging pro Epoche
        if epoch_csv_path is not None:
            self._write_epoch_csv(stage, current_epoch, epoch_csv_path, acc_val, auc_val)

    # ---------------------------
    # CSV-Log pro Epoche
    # ---------------------------
    def _write_epoch_csv(self, stage, epoch, csv_path, acc_val, auc_val):
        if len(self.epoch_losses) == 0:
            return

        total_loss = self.epoch_losses[-1]
        trip_loss  = self.epoch_triplet_losses[-1]
        bce_loss   = self.epoch_bce_losses[-1]

        precisionK = self.metric_history["precision"][-1]
        recallK    = self.metric_history["recall"][-1]
        map_val    = self.metric_history["mAP"][-1]

        row = [
            stage,
            epoch,
            f"{total_loss:.4f}",
            f"{trip_loss:.4f}",
            f"{bce_loss:.4f}",
            f"{precisionK:.4f}",
            f"{recallK:.4f}",
            f"{map_val:.4f}",
            f"{acc_val:.4f}",
            f"{auc_val:.4f}",
        ]

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(row)
        logging.info(f"=> Epoche {epoch} in {csv_path} geloggt (Stage={stage}).")

    # ---------------------------
    # VIS
    # ---------------------------
    def visualize_embeddings(self, df, data_root,
                             method='tsne',
                             epoch=0,
                             output_dir='plots'):
        import os
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

        logging.info(f"Visualisiere Embeddings => {method.upper()}, EPOCH={epoch}")
        embeddings_list = []
        combos = []
        for i, row in df.iterrows():
            pid = row['pid']
            study_yr = row['study_yr']
            combo = row['combination']

            emb = self.compute_patient_embedding(pid, study_yr)
            emb_np = emb.squeeze(0).detach().cpu().numpy()
            embeddings_list.append(emb_np)
            combos.append(combo)

        embeddings_arr = np.array(embeddings_list)
        if method.lower()=='tsne':
            projector = TSNE(n_components=2, random_state=42)
        else:
            projector = PCA(n_components=2, random_state=42)

        coords_2d = projector.fit_transform(embeddings_arr)

        plt.figure(figsize=(8,6))
        unique_combos = sorted(list(set(combos)))
        for c in unique_combos:
            idxs = [ix for ix,val in enumerate(combos) if val==c]
            plt.scatter(coords_2d[idxs,0], coords_2d[idxs,1], label=str(c), alpha=0.6)

        plt.title(f"{method.upper()} Embeddings - EPOCH={epoch}")
        plt.legend(loc='best')

        aggregator_name = getattr(self.mil_agg, '__class__').__name__
        timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fname = os.path.join(
            output_dir,
            f"Emb_{method}_{aggregator_name}_epoch{epoch:03d}_{timestamp}.png"
        )
        plt.savefig(fname, dpi=150)
        plt.close()
        logging.info(f"Saved embedding plot: {fname}")

    # ---------------------------
    # Plot: ACC, AUC vs. Epoch
    # ---------------------------
    def plot_acc_auc_curves(self, output_dir='plots'):
        """
        Zeichnet den Verlauf von Accuracy + AUC vs. Epoche.
        """
        if len(self.binclass_history["acc"]) == 0:
            logging.info("Keine binclass_history => skip plot_acc_auc_curves.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        x_vals = range(1, len(self.binclass_history["acc"])+1)
        acc_vals = self.binclass_history["acc"]
        auc_vals = self.binclass_history["auc"]

        plt.figure(figsize=(8,6))
        plt.plot(x_vals, acc_vals, 'b-o', label='Accuracy')
        plt.plot(x_vals, auc_vals, 'r-^', label='AUC')
        plt.title("Binary Classification: ACC & AUC vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.ylim(0,1)
        plt.legend(loc='best')

        aggregator_name = getattr(self.mil_agg, '__class__').__name__
        timestr = datetime.datetime.now().strftime("%m%d-%H%M")
        outname = os.path.join(output_dir, f"acc_auc_vs_epoch_{aggregator_name}_{timestr}.png")
        plt.savefig(outname, dpi=150)
        plt.close()
        logging.info(f"Saved ACC/AUC curves: {outname}")

    # ---------------------------
    # plot_loss_components, plot_metric_curves
    # ---------------------------
    def plot_loss_components(self, output_dir='plots'):
        if not self.epoch_losses:
            logging.info("Keine Verlaufsdaten => skip plot_loss_components.")
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        epochs_range = range(1, len(self.epoch_losses)+1)

        plt.figure(figsize=(8,6))
        plt.plot(epochs_range, self.epoch_losses,       'r-o', label="Total Loss")
        plt.plot(epochs_range, self.epoch_triplet_losses,'b-^', label="Triplet Loss")
        plt.plot(epochs_range, self.epoch_bce_losses,    'g-s', label="BCE Loss")

        plt.title("Loss Components vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.legend()

        aggregator_name = getattr(self.mil_agg, '__class__').__name__
        timestr = datetime.datetime.now().strftime("%m%d-%H%M")
        outname = os.path.join(output_dir, f"loss_components_{aggregator_name}_{timestr}.png")
        plt.savefig(outname, dpi=150)
        plt.close()
        logging.info(f"Saved loss components plot: {outname}")

    def plot_metric_curves(self, output_dir='plots'):
        if len(self.metric_history["mAP"])==0:
            logging.info("Keine IR-Metriken => skip plot_metric_curves.")
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        x_vals = list(range(1, len(self.metric_history["mAP"])+1))

        plt.figure(figsize=(8,6))
        plt.plot(x_vals, self.metric_history["precision"], 'g-o', label='Precision@K')
        plt.plot(x_vals, self.metric_history["recall"],    'm-s', label='Recall@K')
        plt.plot(x_vals, self.metric_history["mAP"],       'r-^', label='mAP')
        plt.ylim(0,1)
        plt.title("Precision@K, Recall@K, mAP vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()

        aggregator_name = getattr(self.mil_agg, '__class__').__name__
        timestr = datetime.datetime.now().strftime("%m%d-%H%M")
        outname = os.path.join(output_dir, f"metrics_vs_epoch_{aggregator_name}_{timestr}.png")
        plt.savefig(outname, dpi=150)
        plt.close()
        logging.info(f"Saved IR metric curves: {outname}")

    # ---------------------------
    # Checkpoints
    # ---------------------------
    def save_checkpoint(self, path):
        ckpt = {
            'base_cnn': self.base_cnn.state_dict(),
            'mil_agg': self.mil_agg.state_dict(),
            'classifier': self.classifier.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if self.scheduler is not None:
            ckpt['scheduler'] = self.scheduler.state_dict()
        torch.save(ckpt, path)
        logging.info(f"Checkpoint saved => {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.base_cnn.load_state_dict(ckpt['base_cnn'])
        self.mil_agg.load_state_dict(ckpt['mil_agg'])
        self.classifier.load_state_dict(ckpt['classifier'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

        if self.scheduler is not None and 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])

        logging.info(f"Checkpoint loaded from {path}")
