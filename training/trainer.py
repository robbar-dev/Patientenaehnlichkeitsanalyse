import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import datetime

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


# ------------------------------------------------
# (1) Funktion zum Parsen: "1-0-0" => 1 (krank), "0-0-1" => 0 (gesund)
# ------------------------------------------------
def parse_combo_str_to_vec(combo_str):
    """
    Für deinen neuen Datensatz hast du im CSV-Feld 'combination' entweder '1-0-0' oder '0-0-1'.
    Wir mappen das hier auf einen einzelnen Integer:
      '1-0-0' => 1 (krank)
      '0-0-1' => 0 (gesund)

    Falls du mehr Varianten hättest, könntest du hier zusätzliche if-Abfragen ergänzen.
    """
    parts = [int(x) for x in combo_str.split('-')]
    if parts == [1,0,0]:
        return 1
    else:
        # Annahme: Alle anderen Fälle (insbesondere '0-0-1') => 0
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
        # SinglePatientDataset
        skip_slices=True,
        skip_factor=2,
        filter_empty_patches=False,
        min_nonzero_fraction=0.01,
        filter_uniform_patches=True,
        min_std_threshold=0.01,
        do_patch_minmax=False,
        # Hybrid-Loss
        lambda_bce=1.0,
        # NEU: manueller Parameter für Augmentation im Training
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

        # (D) Binary-Kopf + BCE-Loss
        #    => Nur 1 Ausgabe-Logit (gesund vs. krank)
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

        # Verlaufslisten
        self.epoch_losses = []         # Gesamtloss (Triplet + BCE)
        self.epoch_triplet_losses = [] # Nur Triplet
        self.epoch_bce_losses = []     # Nur BCE

        self.metric_history = {
            "precision": [],
            "recall": [],
            "mAP": []
        }

        # Optional für das bisherige Multi-Label-Tracking
        # (wir füllen das weiterhin, damit deine Plots nicht crashen)
        self.multilabel_history = {
            "fibrose_f1": [],
            "emphysem_f1": [],
            "nodule_f1": [],
            "macro_f1": []
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
        """
        => Nur beim Training. do_augmentation_train steuert Augmentation.
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

        loader = DataLoader(ds, batch_size=32, shuffle=False)

        # Sicherstellen, dass wir im Train-Modus sind (BatchNorm/Dropout etc.)
        self.base_cnn.train()
        self.mil_agg.train()
        self.classifier.train()

        patch_embs = []
        for patch_t in loader:
            patch_t = patch_t.to(self.device)
            emb = self.base_cnn(patch_t)
            patch_embs.append(emb)

        if len(patch_embs) == 0:
            # Edge-Case: Falls kein Patch existiert, nimm Dummy
            dummy = torch.zeros((1,512), device=self.device, requires_grad=True)
            dummy_logits = self.classifier(dummy)
            return dummy, dummy_logits

        patch_embs = torch.cat(patch_embs, dim=0)
        patient_emb = self.mil_agg(patch_embs)      # z.B. (1,512)
        logits = self.classifier(patient_emb)       # => (1,1)
        return patient_emb, logits

    def compute_patient_embedding(self, pid, study_yr):
        """
        => Validation/Test: keine Augmentation
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

            # Früher "multi_label", jetzt nur label=0/1
            a_label = anchor_info['label']
            p_label = pos_info['label']
            n_label = neg_info['label']

            # Forward => Aug = True
            a_emb, a_logits = self._forward_patient(a_pid, a_sy)
            p_emb, p_logits = self._forward_patient(p_pid, p_sy)
            n_emb, n_logits = self._forward_patient(n_pid, n_sy)

            # (1) TripletLoss
            trip_loss = self.triplet_loss_fn(a_emb, p_emb, n_emb)

            # (2) BCE-Loss
            #     Wir wandeln int=> Tensor shape (1,1)
            a_label_t = torch.tensor([[a_label]], dtype=torch.float32, device=self.device)
            p_label_t = torch.tensor([[p_label]], dtype=torch.float32, device=self.device)
            n_label_t = torch.tensor([[n_label]], dtype=torch.float32, device=self.device)

            bce_a = self.bce_loss_fn(a_logits, a_label_t)
            bce_p = self.bce_loss_fn(p_logits, p_label_t)
            bce_n = self.bce_loss_fn(n_logits, n_label_t)
            bce_loss = (bce_a + bce_p + bce_n) / 3.0

            # (3) Gesamt-Loss
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
        # Beispielhafter HardNegative-Sampler
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
    # train_with_val => + evaluate
    # ---------------------------
    def train_with_val(
        self,
        epochs,
        num_triplets,
        val_csv,
        data_root_val,
        K=10,
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

        df_val = pd.read_csv(val_csv)

        for epoch in range(1, epochs+1):
            logging.info(f"=== EPOCH {epoch}/{epochs} ===")
            # 1 Epoche train
            self.train_loop(num_epochs=1, num_triplets=num_triplets)

            # (2) Evaluate => IR-Metriken
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

            # Track best IR-mAP
            if map_val > self.best_val_map:
                self.best_val_map = map_val
                self.best_val_epoch = epoch
                logging.info(f"=> New Best mAP={map_val:.4f} @ epoch={epoch}")

            # (3) Evaluate => "Multi-Label"-Funktion, jetzt binär getrimmt
            multilabel_results = self.evaluate_multilabel_classification(
                df_val,
                data_root_val,
                threshold=0.5
            )
            logging.info(f"Multi-Label => {multilabel_results}")

            self.multilabel_history["fibrose_f1"].append(multilabel_results["fibrose_f1"])
            self.multilabel_history["emphysem_f1"].append(multilabel_results["emphysem_f1"])
            self.multilabel_history["nodule_f1"].append(multilabel_results["nodule_f1"])
            self.multilabel_history["macro_f1"].append(multilabel_results["macro_f1"])

            # (4) Visualisierung
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
        self.plot_multilabel_f1_curves(output_dir=output_dir)

        return (self.best_val_map, self.best_val_epoch)


    # ---------------------------
    # train_with_val_2stage (Hard Negatives)
    # ---------------------------
    def train_with_val_2stage(
        self,
        epochs_stage1,
        epochs_stage2,
        num_triplets,
        val_csv,
        data_root_val,
        K=10,
        distance_metric='euclidean',
        visualize_every=5,
        visualize_method='tsne',
        output_dir='plots'
    ):
        df_val = pd.read_csv(val_csv)

        # Reset Tracking
        self.best_val_map = 0.0
        self.best_val_epoch = -1
        self.epoch_losses = []
        self.epoch_triplet_losses = []
        self.epoch_bce_losses = []
        self.metric_history = {"precision": [], "recall": [], "mAP": []}
        self.multilabel_history = {
            "fibrose_f1": [],
            "emphysem_f1": [],
            "nodule_f1": [],
            "macro_f1": []
        }

        total_epochs = epochs_stage1 + epochs_stage2
        current_epoch = 0

        # 1) Stage1 => normal Sampler
        normal_sampler = TripletSampler(df=self.df, num_triplets=num_triplets,
                                        shuffle=True, top_k_negatives=3)

        for e in range(1, epochs_stage1+1):
            current_epoch += 1
            logging.info(f"=== STAGE1-Epoch {current_epoch}/{total_epochs} ===")
            self._train_one_epoch_internal(normal_sampler)
            normal_sampler.reset_epoch()

            # Evaluate (IR) + Multi-Label
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
                output_dir=output_dir
            )

        # 2) Stage2 => HardNegative
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

            # Evaluate (IR) + Multi-Label
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
                output_dir=output_dir
            )

        self.plot_loss_components(output_dir=output_dir)
        self.plot_metric_curves(output_dir=output_dir)
        self.plot_multilabel_f1_curves(output_dir=output_dir)

        return self.best_val_map, self.best_val_epoch


    # ---------------------------
    # Hilfsfunktion für Evaluate + Visuals
    # ---------------------------
    def _evaluate_and_visualize(
        self,
        current_epoch,
        total_epochs,
        val_csv,
        df_val,
        data_root_val,
        K=10,
        distance_metric='euclidean',
        visualize_every=5,
        visualize_method='tsne',
        output_dir='plots'
    ):
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

        # Multi-Label (jetzt binär) => wir belassen den Funktionsnamen
        # und loggen in das vorhandene dict.
        multilabel_results = self.evaluate_multilabel_classification(df_val, data_root_val, threshold=0.5)
        logging.info(f"Multi-Label => {multilabel_results}")

        self.multilabel_history["fibrose_f1"].append(multilabel_results["fibrose_f1"])
        self.multilabel_history["emphysem_f1"].append(multilabel_results["emphysem_f1"])
        self.multilabel_history["nodule_f1"].append(multilabel_results["nodule_f1"])
        self.multilabel_history["macro_f1"].append(multilabel_results["macro_f1"])

        # Visualisierung
        if current_epoch % visualize_every == 0:
            self.visualize_embeddings(
                df=df_val,
                data_root=data_root_val,
                method=visualize_method,
                epoch=current_epoch,
                output_dir=output_dir
            )


    # ---------------------------
    # evaluate_multilabel_classification => minimal auf Binärlogik getrimmt
    # ---------------------------
    def evaluate_multilabel_classification(self, df, data_root, threshold=0.5):
        """
        Ehemals Multi-Label, jetzt nur noch 0/1:
          - parse_combo_str_to_vec => 0/1
          - BCE => Wir berechnen TP,FP,FN nur für "krank=1".
          - Die Dictionary-Keys fibrose/emphysem/nodule behalten wir bei,
            damit der restliche Code/Plot nicht abstürzt.
        """
        self.base_cnn.eval()
        self.mil_agg.eval()
        self.classifier.eval()

        TP = 0
        FP = 0
        FN = 0

        for idx, row in df.iterrows():
            pid = row['pid']
            study_yr = row['study_yr']
            combo_str = row['combination']

            gt_label = parse_combo_str_to_vec(combo_str)  # => 0 oder 1

            with torch.no_grad():
                emb = self.compute_patient_embedding(pid, study_yr)
                logits = self.classifier(emb)    # shape (1,1)
                prob = torch.sigmoid(logits).item()
                pred = 1 if prob >= threshold else 0

            if gt_label == 1 and pred == 1:
                TP += 1
            elif gt_label == 0 and pred == 1:
                FP += 1
            elif gt_label == 1 and pred == 0:
                FN += 1

        precision = TP/(TP+FP) if (TP+FP)>0 else 0
        recall    = TP/(TP+FN) if (TP+FN)>0 else 0
        f1 = 0
        if precision+recall > 0:
            f1 = 2*precision*recall/(precision+recall)

        # Wir "missbrauchen" die alten Keys und tragen 0 für unnötige Felder ein.
        results = {}
        results["fibrose_precision"] = precision
        results["fibrose_recall"]    = recall
        results["fibrose_f1"]        = f1
        results["emphysem_f1"]       = 0
        results["nodule_f1"]         = 0
        results["macro_f1"]          = f1   # Macro-F1 = F1 bei 2-Klassen

        return results


    # ---------------------------
    # visualize_embeddings (unverändert)
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
    # plot_loss_components, plot_metric_curves, ...
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
        logging.info(f"Saved metric curves: {outname}")

    def plot_multilabel_f1_curves(self, output_dir='plots'):
        if not hasattr(self, 'multilabel_history'):
            logging.info("No multilabel_history => skip plot_multilabel_f1_curves.")
            return
        if len(self.multilabel_history["macro_f1"]) == 0:
            logging.info("No data in multilabel_history => skip plot_multilabel_f1_curves.")
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        x_vals = range(1, len(self.multilabel_history["macro_f1"])+1)

        fib_f1   = self.multilabel_history["fibrose_f1"]
        emph_f1  = self.multilabel_history["emphysem_f1"]
        nod_f1   = self.multilabel_history["nodule_f1"]
        mac_f1   = self.multilabel_history["macro_f1"]

        plt.figure(figsize=(8,6))
        plt.plot(x_vals, fib_f1,   'r-o', label='Fibrose F1')
        plt.plot(x_vals, emph_f1,  'g-s', label='Emphysem F1')
        plt.plot(x_vals, nod_f1,   'b-^', label='Nodule F1')
        plt.plot(x_vals, mac_f1,   'k--', label='Macro-F1', linewidth=2.0)

        plt.title("Multi-Label F1 vs. Epoch (eigentlich Binär)")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.ylim(0,1)
        plt.legend(loc='best')

        aggregator_name = getattr(self.mil_agg, '__class__').__name__
        timestr = datetime.datetime.now().strftime("%m%d-%H%M")
        outname = os.path.join(output_dir, f"multilabel_f1_vs_epoch_{aggregator_name}_{timestr}.png")

        plt.savefig(outname, dpi=150)
        plt.close()
        logging.info(f"Saved multi-label F1 curve: {outname}")

    # ---------------------------
    # Checkpoint
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
