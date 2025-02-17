import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import datetime
import csv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

def parse_combo_str_to_vec(combo_str):
    """ '0-1-1' => [0,1,1] """
    return [int(x) for x in combo_str.split('-')]


class TripletTrainer(nn.Module):
    """
    Trainer expanded architecture:
     - ResNet18/50 (BaseCNN, param: model_name, freeze_blocks)
     - Attention-MIL / Max / Mean aggregator (param aggregator_name)
     - TripletMarginLoss + BCE (Multi-Label)
     - Hard Negative Mining (2-stage)
     - Evaluate => IR (precision@K, recall@K, mAP)
     - Evaluate => Multi-Label (3-Klassen: fibrose, emphysem, nodule)
     - CSV-Logging
    """

    def __init__(
        self,
        df,
        data_root,
        device='cuda',
        lr=1e-4,
        margin=1.0,
        model_name='resnet18',
        freeze_blocks=None,
        aggregator_name='mil',
        agg_hidden_dim=128,
        agg_dropout=0.2,
        roi_size=(96,96,3),
        overlap=(10,10,1),
        do_augmentation=False,
        lambda_bce=1.0  # gewichte BCE vs. Triplet
    ):
        super().__init__()
        self.df = df
        self.data_root = data_root
        self.device = device
        self.lr = lr
        self.margin = margin
        self.model_name = model_name
        self.freeze_blocks = freeze_blocks
        self.aggregator_name = aggregator_name
        self.agg_hidden_dim = agg_hidden_dim
        self.agg_dropout = agg_dropout
        self.roi_size = roi_size
        self.overlap = overlap
        self.do_augmentation = do_augmentation
        self.lambda_bce = lambda_bce

        # CNN-Backbone
        from model.base_cnn import BaseCNN
        self.base_cnn = BaseCNN(
            model_name=self.model_name,
            pretrained=True,
            freeze_blocks=self.freeze_blocks
        ).to(device)

        # Aggregator je nach aggregator_name
        from model.mil_aggregator import AttentionMILAggregator
        from model.max_pooling import MaxPoolingAggregator
        from model.mean_pooling import MeanPoolingAggregator

        if aggregator_name == 'mil':
            self.mil_agg = AttentionMILAggregator(
                in_dim=512,
                hidden_dim=self.agg_hidden_dim,
                dropout=self.agg_dropout
            ).to(device)
        elif aggregator_name == 'max':
            self.mil_agg = MaxPoolingAggregator().to(device)
        elif aggregator_name == 'mean':
            self.mil_agg = MeanPoolingAggregator().to(device)
        else:
            raise ValueError(f"Unbekannte aggregator_name={aggregator_name}. Nutze 'mil','max','mean'.")

        # TripletLoss
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2)

        # Multi-Label-Kopf => 3-Klassen => BCE
        self.classifier = nn.Linear(512, 3).to(device)
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

        # Optimizer
        params = list(self.base_cnn.parameters()) \
               + list(self.mil_agg.parameters()) \
               + list(self.classifier.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr)

        self.epoch_losses = []          
        self.epoch_triplet_losses = [] 
        self.epoch_bce_losses = []    
        self.metric_history = {
            "precision": [],
            "recall": [],
            "mAP": []
        }
        self.multilabel_history = {
            "fibrose_f1": [],
            "emphysem_f1": [],
            "nodule_f1": [],
            "macro_f1": []
        }

        self.best_val_map = 0.0
        self.best_val_epoch = -1

        logging.info("[TripletTrainer] Initialized")
        logging.info(f"lr={lr}, margin={margin}, aggregator={aggregator_name}, "
                     f"freeze_blocks={freeze_blocks}, agg_hidden_dim={agg_hidden_dim}, "
                     f"agg_dropout={agg_dropout}, do_augmentation={do_augmentation}, "
                     f"lambda_bce={lambda_bce}")

    def _forward_patient(self, pid, study_yr, do_train=True):
        """
        Lädt SinglePatientDataset => Patches => base_cnn => mil_agg => (1,512).
        do_train steuert, ob Augmentation und train() Modus oder eval() => no Augmen
        """
        from training.data_loader import SinglePatientDataset
        ds = SinglePatientDataset(
            data_root=self.data_root,
            pid=pid,
            study_yr=study_yr,
            roi_size=self.roi_size,
            overlap=self.overlap,
            skip_factor=2,
            do_augmentation=(self.do_augmentation if do_train else False)
        )
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        if do_train:
            self.base_cnn.train()
            self.mil_agg.train()
            self.classifier.train()
        else:
            self.base_cnn.eval()
            self.mil_agg.eval()
            self.classifier.eval()

        patch_embs = []
        for patch_t in loader:
            patch_t = patch_t.to(self.device)
            emb = self.base_cnn(patch_t)  # => (B,512)
            patch_embs.append(emb)

        if len(patch_embs)==0:
            # Edge case
            dummy = torch.zeros((1,512), device=self.device, requires_grad=do_train)
            logits = self.classifier(dummy)
            return dummy, logits

        patch_embs = torch.cat(patch_embs, dim=0)
        patient_emb = self.mil_agg(patch_embs)
        logits = self.classifier(patient_emb)  # => shape (1,3)
        return patient_emb, logits

    def compute_patient_embedding(self, pid, study_yr):
        """
        Für Evaluate => do_train=False => no augmentation
        => returns (1,512) patient_emb
        """
        emb, logits = self._forward_patient(pid, study_yr, do_train=False)
        return emb  # (1,512)

    def train_one_epoch(self, sampler):
        """
        Sampler => (anchor_info, pos_info, neg_info)
        -> multi_label = [0,1,1]
        => Triplet + BCE
        """
        total_loss = 0.0
        total_trip = 0.0
        total_bce  = 0.0
        steps = 0

        for step, (a_info, p_info, n_info) in enumerate(sampler):
            # anchor/pos/neg => pid, study_yr, multi_label
            a_pid, a_sy = a_info['pid'], a_info['study_yr']
            p_pid, p_sy = p_info['pid'], p_info['study_yr']
            n_pid, n_sy = n_info['pid'], n_info['study_yr']

            a_label = a_info['multi_label']  # e.g. [0,1,1]
            p_label = p_info['multi_label']
            n_label = n_info['multi_label']

            a_emb, a_logits = self._forward_patient(a_pid, a_sy, do_train=True)
            p_emb, p_logits = self._forward_patient(p_pid, p_sy, do_train=True)
            n_emb, n_logits = self._forward_patient(n_pid, n_sy, do_train=True)

            trip_loss = self.triplet_loss_fn(a_emb, p_emb, n_emb)

            # BCE =>  shape(1,3)
            a_label_t = torch.tensor(a_label, dtype=torch.float32, device=self.device).unsqueeze(0)
            p_label_t = torch.tensor(p_label, dtype=torch.float32, device=self.device).unsqueeze(0)
            n_label_t = torch.tensor(n_label, dtype=torch.float32, device=self.device).unsqueeze(0)

            bce_a = self.bce_loss_fn(a_logits, a_label_t)
            bce_p = self.bce_loss_fn(p_logits, p_label_t)
            bce_n = self.bce_loss_fn(n_logits, n_label_t)
            bce_loss = (bce_a + bce_p + bce_n)/3.0

            loss = trip_loss + self.lambda_bce * bce_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_trip += trip_loss.item()
            total_bce  += bce_loss.item()
            steps += 1

            if step%250 == 0:
                logging.info(f"[Step {step}] TotalLoss={loss.item():.4f}, Trip={trip_loss.item():.4f}, BCE={bce_loss.item():.4f}")

        if steps>0:
            avg_loss = total_loss/steps
            avg_trip = total_trip/steps
            avg_bce  = total_bce/steps
        else:
            avg_loss=0; avg_trip=0; avg_bce=0

        self.epoch_losses.append(avg_loss)
        self.epoch_triplet_losses.append(avg_trip)
        self.epoch_bce_losses.append(avg_bce)

        logging.info(f"=> EPOCH Loss={avg_loss:.4f}, Trip={avg_trip:.4f}, BCE={avg_bce:.4f}")

    def visualize_embeddings(self, df, data_root, method='tsne',
                         epoch=0, output_dir='plots'):
        logging.info(f"Visualisiere Embeddings => {method.upper()}, EPOCH={epoch}")
        embeddings_list = []
        combos = []
        for i, row in df.iterrows():
            pid = row['pid']
            study_yr = row['study_yr']
            combo = row['combination']

            # => compute embedding (1,512)
            emb = self.compute_patient_embedding(pid, study_yr)
            emb_np = emb.squeeze(0).detach().cpu().numpy()
            embeddings_list.append(emb_np)
            combos.append(combo)

        embeddings_arr = np.array(embeddings_list)

        projector = TSNE(n_components=2, random_state=42)

        coords_2d = projector.fit_transform(embeddings_arr)

        plt.figure(figsize=(8,6))
        unique_combos = sorted(list(set(combos)))
        for c in unique_combos:
            idxs = [ix for ix,val in enumerate(combos) if val==c]
            plt.scatter(coords_2d[idxs,0], coords_2d[idxs,1], label=str(c), alpha=0.6)

        plt.title(f"t-SNE Embeddings - EPOCH={epoch}")
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

    # Hard Negative => train_one_epoch_internal 
    def _train_one_epoch_internal(self, sampler):
        total_loss = 0.0
        total_trip = 0.0
        total_bce  = 0.0
        steps = 0

        for step, (a_info, p_info, n_info) in enumerate(sampler):
            a_pid, a_sy = a_info['pid'], a_info['study_yr']
            p_pid, p_sy = p_info['pid'], p_info['study_yr']
            n_pid, n_sy = n_info['pid'], n_info['study_yr']

            a_label = a_info['multi_label']
            p_label = p_info['multi_label']
            n_label = n_info['multi_label']

            a_emb, a_logits = self._forward_patient(a_pid, a_sy, do_train=True)
            p_emb, p_logits = self._forward_patient(p_pid, p_sy, do_train=True)
            n_emb, n_logits = self._forward_patient(n_pid, n_sy, do_train=True)

            trip_loss = self.triplet_loss_fn(a_emb, p_emb, n_emb)

            a_label_t = torch.tensor(a_label, dtype=torch.float32, device=self.device).unsqueeze(0)
            p_label_t = torch.tensor(p_label, dtype=torch.float32, device=self.device).unsqueeze(0)
            n_label_t = torch.tensor(n_label, dtype=torch.float32, device=self.device).unsqueeze(0)

            bce_a = self.bce_loss_fn(a_logits, a_label_t)
            bce_p = self.bce_loss_fn(p_logits, p_label_t)
            bce_n = self.bce_loss_fn(n_logits, n_label_t)
            bce_loss = (bce_a + bce_p + bce_n)/3.0

            loss = trip_loss + self.lambda_bce*bce_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_trip += trip_loss.item()
            total_bce  += bce_loss.item()
            steps += 1

            if step % 250==0:
                logging.info(f"[Step {step}] TotalLoss={loss.item():.4f}, Trip={trip_loss.item():.4f}, BCE={bce_loss.item():.4f}")

        if steps>0:
            avg_loss = total_loss/steps
            avg_trip = total_trip/steps
            avg_bce  = total_bce/steps
        else:
            avg_loss=0; avg_trip=0; avg_bce=0

        self.epoch_losses.append(avg_loss)
        self.epoch_triplet_losses.append(avg_trip)
        self.epoch_bce_losses.append(avg_bce)

        logging.info(f"=> EPOCH Loss={avg_loss:.4f}, Trip={avg_trip:.4f}, BCE={avg_bce:.4f}")

    # train_loop mit normalem Sampler
    def train_loop(self, num_epochs=5, num_triplets=100):
        """
        Simpler Loop => normal Sampler
        """
        from training.triplet_sampler import TripletSampler
        sampler = TripletSampler(
            df=self.df,
            num_triplets=num_triplets,
            shuffle=True,
            top_k_negatives=3
        )

        for epoch in range(1, num_epochs+1):
            logging.info(f"=== EPOCH {epoch}/{num_epochs} ===")
            self.train_one_epoch(sampler)
            sampler.reset_epoch()

    # Ab hier die Implementierung des 2-Stage Trainings mit Hard & Normal Sampler
    # train_with_val => 1-stage
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
        output_dir='plots',
        epoch_csv_path=None
    ):
        # Reset
        self.epoch_losses = []
        self.epoch_triplet_losses = []
        self.epoch_bce_losses = []
        self.metric_history = {"precision": [], "recall": [], "mAP": []}
        self.multilabel_history = {"fibrose_f1": [], "emphysem_f1": [], "nodule_f1": [], "macro_f1": []}
        self.best_val_map = 0.0
        self.best_val_epoch = -1

        df_val = pd.read_csv(val_csv)

        if epoch_csv_path and not os.path.exists(epoch_csv_path):
            with open(epoch_csv_path, mode='w', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                header = [
                    "Epoch","TripletLoss","BCE_Loss","Precision@K","Recall@K","mAP",
                    "fibrose_f1","emphysem_f1","nodule_f1","macro_f1"
                ]
                writer.writerow(header)

        from training.triplet_sampler import TripletSampler
        from evaluation.metrics import compute_embeddings, compute_precision_recall_map

        sampler = TripletSampler(
            df=self.df,
            num_triplets=num_triplets,
            shuffle=True,
            top_k_negatives=3
        )

        for epoch in range(1, epochs+1):
            logging.info(f"=== EPOCH {epoch}/{epochs} ===")
            self.train_one_epoch(sampler)
            sampler.reset_epoch()

            # Evaluate IR
            emb_dict = compute_embeddings(
                trainer=self,
                df=df_val,
                data_root=data_root_val,
                device=self.device
            )
            val_metrics = compute_precision_recall_map(emb_dict, K=K, distance_metric=distance_metric)
            precK = val_metrics["precision@K"]
            recK  = val_metrics["recall@K"]
            map_val = val_metrics["mAP"]

            logging.info(f"[Val-Epoch={epoch}] P@K={precK:.4f}, R@K={recK:.4f}, mAP={map_val:.4f}")
            self.metric_history["precision"].append(precK)
            self.metric_history["recall"].append(recK)
            self.metric_history["mAP"].append(map_val)

            if map_val>self.best_val_map:
                self.best_val_map = map_val
                self.best_val_epoch = epoch
                logging.info(f"=> New Best mAP={map_val:.4f} @ epoch={epoch}")

            # Evaluate Multi-Label
            ml_res = self.evaluate_multilabel_classification(df_val, data_root_val, threshold=0.5)
            fib_f1 = ml_res["fibrose_f1"]
            emph_f1= ml_res["emphysem_f1"]
            nod_f1 = ml_res["nodule_f1"]
            mac_f1 = ml_res["macro_f1"]
            logging.info(f"MultiLabel => {ml_res}")

            self.multilabel_history["fibrose_f1"].append(fib_f1)
            self.multilabel_history["emphysem_f1"].append(emph_f1)
            self.multilabel_history["nodule_f1"].append(nod_f1)
            self.multilabel_history["macro_f1"].append(mac_f1)

            # Visualization
            if epoch % visualize_every==0:
                self.visualize_embeddings(
                    df=df_val,
                    data_root=data_root_val,
                    method=visualize_method,
                    epoch=epoch,
                    output_dir=output_dir
                )

            # CSV
            if epoch_csv_path:
                self._write_epoch_csv_ml(epoch, epoch_csv_path, precK, recK, map_val, fib_f1, emph_f1, nod_f1, mac_f1)

        self.plot_loss_components(output_dir=output_dir)
        self.plot_metric_curves(output_dir=output_dir)
        self.plot_multilabel_f1_curves(output_dir=output_dir)

        return self.best_val_map, self.best_val_epoch

    def _write_epoch_csv_ml(self, epoch, csv_path, precK, recK, map_val, fib_f1, emph_f1, nod_f1, mac_f1):
        if len(self.epoch_triplet_losses)==0 or len(self.epoch_bce_losses)==0:
            return
        trip_loss = self.epoch_triplet_losses[-1]
        bce_loss  = self.epoch_bce_losses[-1]

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            row = [
                epoch,
                f"{trip_loss:.4f}",
                f"{bce_loss:.4f}",
                f"{precK:.4f}",
                f"{recK:.4f}",
                f"{map_val:.4f}",
                f"{fib_f1:.4f}",
                f"{emph_f1:.4f}",
                f"{nod_f1:.4f}",
                f"{mac_f1:.4f}"
            ]
            writer.writerow(row)
        logging.info(f"=> Epoche {epoch} in {csv_path} geloggt.")

    # 2-stage => HardNegative
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
        output_dir='plots',
        epoch_csv_path=None
    ):
        """
        Stage1 => normal sampler
        Stage2 => HardNegative => neu berechnen alle 5 Epochen
        => Evaluate IR + Multi-Label
        => Plots
        """
        df_val = pd.read_csv(val_csv)

        # Reset
        self.epoch_losses = []
        self.epoch_triplet_losses = []
        self.epoch_bce_losses = []
        self.metric_history = {"precision": [], "recall": [], "mAP": []}
        self.multilabel_history = {"fibrose_f1": [], "emphysem_f1": [], "nodule_f1": [], "macro_f1": []}
        self.best_val_map = 0.0
        self.best_val_epoch = -1

        if epoch_csv_path and not os.path.exists(epoch_csv_path):
            with open(epoch_csv_path, mode='w', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                header = [
                    "Stage","Epoch","TripletLoss","BCE_Loss",
                    "Precision@K","Recall@K","mAP",
                    "fibrose_f1","emphysem_f1","nodule_f1","macro_f1"
                ]
                writer.writerow(header)

        total_epochs = epochs_stage1 + epochs_stage2
        current_epoch = 0

        # Stage1 => normal
        from training.triplet_sampler import TripletSampler
        sampler_normal = TripletSampler(
            df=self.df,
            num_triplets=num_triplets,
            shuffle=True,
            top_k_negatives=3
        )

        for e in range(1, epochs_stage1+1):
            current_epoch += 1
            stage_name = "Stage1"
            logging.info(f"=== STAGE1-Epoch {current_epoch}/{total_epochs} ===")

            self._train_one_epoch_internal(sampler_normal)
            sampler_normal.reset_epoch()

            # Evaluate
            self._evaluate_and_visualize(current_epoch, total_epochs, df_val, val_csv,
                                         data_root_val, K, distance_metric,
                                         visualize_every, visualize_method, output_dir,
                                         stage_name, epoch_csv_path)

        # Stage2 => Hard Neg
        for e in range(1, epochs_stage2+1):
            current_epoch += 1
            stage_name = "Stage2"
            logging.info(f"=== STAGE2-Epoch {current_epoch}/{total_epochs} ===")

            if (e % 5)==1:
                from training.triplet_sampler_hard_negative import HardNegativeTripletSampler
                hard_sampler = HardNegativeTripletSampler(
                    df=self.df,
                    trainer=self,
                    num_triplets=num_triplets,
                    device=self.device
                )
            self._train_one_epoch_internal(hard_sampler)

            # Evaluate
            self._evaluate_and_visualize(current_epoch, total_epochs, df_val, val_csv,
                                         data_root_val, K, distance_metric,
                                         visualize_every, visualize_method, output_dir,
                                         stage_name, epoch_csv_path)

        self.plot_loss_components(output_dir=output_dir)
        self.plot_metric_curves(output_dir=output_dir)
        self.plot_multilabel_f1_curves(output_dir=output_dir)

        return self.best_val_map, self.best_val_epoch

    def _evaluate_and_visualize(
        self,
        current_epoch,
        total_epochs,
        df_val,
        val_csv,
        data_root_val,
        K,
        distance_metric,
        visualize_every,
        visualize_method,
        output_dir,
        stage_name,
        epoch_csv_path
    ):
        # IR
        from evaluation.metrics import compute_embeddings, compute_precision_recall_map
        emb_dict = compute_embeddings(
            trainer=self,
            df=df_val,
            data_root=data_root_val,
            device=self.device
        )
        val_metrics = compute_precision_recall_map(emb_dict, K=K, distance_metric=distance_metric)
        precK = val_metrics["precision@K"]
        recK  = val_metrics["recall@K"]
        map_val= val_metrics["mAP"]

        self.metric_history["precision"].append(precK)
        self.metric_history["recall"].append(recK)
        self.metric_history["mAP"].append(map_val)

        if map_val>self.best_val_map:
            self.best_val_map = map_val
            self.best_val_epoch = current_epoch
            logging.info(f"=> New Best mAP={map_val:.4f} @ epoch={current_epoch}")

        logging.info(f"[Val-Epoch={current_epoch}] P@K={precK:.4f}, R@K={recK:.4f}, mAP={map_val:.4f}")

        # Multi-Label
        ml_res = self.evaluate_multilabel_classification(df_val, data_root_val, threshold=0.5)
        fib_f1 = ml_res["fibrose_f1"]
        emph_f1= ml_res["emphysem_f1"]
        nod_f1 = ml_res["nodule_f1"]
        mac_f1 = ml_res["macro_f1"]
        self.multilabel_history["fibrose_f1"].append(fib_f1)
        self.multilabel_history["emphysem_f1"].append(emph_f1)
        self.multilabel_history["nodule_f1"].append(nod_f1)
        self.multilabel_history["macro_f1"].append(mac_f1)

        logging.info(f"MultiLabel => {ml_res}")

        # Visualization
        if current_epoch%visualize_every==0:
            self.visualize_embeddings(
                df=df_val,
                data_root=data_root_val,
                method=visualize_method,
                epoch=current_epoch,
                output_dir=output_dir
            )

        if epoch_csv_path:
            self._write_epoch_csv_2stage(
                stage=stage_name,
                epoch=current_epoch,
                csv_path=epoch_csv_path,
                precK=precK, recK=recK, map_val=map_val,
                fib_f1=fib_f1, emph_f1=emph_f1, nod_f1=nod_f1, mac_f1=mac_f1
            )

    def _write_epoch_csv_2stage(self, stage, epoch, csv_path,
                                precK, recK, map_val,
                                fib_f1, emph_f1, nod_f1, mac_f1):
        if len(self.epoch_triplet_losses)==0 or len(self.epoch_bce_losses)==0:
            return
        trip = self.epoch_triplet_losses[-1]
        bce  = self.epoch_bce_losses[-1]
        row = [
            stage, epoch,
            f"{trip:.4f}",
            f"{bce:.4f}",
            f"{precK:.4f}",
            f"{recK:.4f}",
            f"{map_val:.4f}",
            f"{fib_f1:.4f}",
            f"{emph_f1:.4f}",
            f"{nod_f1:.4f}",
            f"{mac_f1:.4f}"
        ]
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(row)
        logging.info(f"=> Epoche {epoch} (Stage={stage}) in CSV geloggt => {csv_path}")

    # Evaluate Multi-Label
    def evaluate_multilabel_classification(self, df, data_root, threshold=0.5):
        self.base_cnn.eval()
        self.mil_agg.eval()
        self.classifier.eval()

        TP = [0,0,0]
        FP = [0,0,0]
        FN = [0,0,0]

        for idx, row in df.iterrows():
            pid = row['pid']
            study_yr = row['study_yr']
            combo_str = row['combination']
            gt_vec = parse_combo_str_to_vec(combo_str) 

            # Emb + Logits + Sigmoid
            with torch.no_grad():
                emb, logits = self._forward_patient(pid, study_yr, do_train=False)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                pred  = [1 if p>=threshold else 0 for p in probs]

            for i in range(3):
                if gt_vec[i]==1 and pred[i]==1:
                    TP[i]+=1
                elif gt_vec[i]==0 and pred[i]==1:
                    FP[i]+=1
                elif gt_vec[i]==1 and pred[i]==0:
                    FN[i]+=1

        names = ["fibrose","emphysem","nodule"]
        results = {}
        sum_f1 = 0
        for i,name in enumerate(names):
            prec = TP[i]/(TP[i]+FP[i]) if (TP[i]+FP[i])>0 else 0
            rec  = TP[i]/(TP[i]+FN[i]) if (TP[i]+FN[i])>0 else 0
            f1_i = 0
            if prec+rec>0:
                f1_i = 2*prec*rec/(prec+rec)
            results[f"{name}_precision"] = prec
            results[f"{name}_recall"]    = rec
            results[f"{name}_f1"]        = f1_i
            sum_f1+= f1_i

        results["macro_f1"] = sum_f1/3.0
        return results

    # Plotting
    def plot_loss_components(self, output_dir='plots'):
        import matplotlib.pyplot as plt
        import os

        if len(self.epoch_losses)==0:
            logging.info("Keine Verlaufsdaten => skip plot_loss_components.")
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        x_vals = range(1, len(self.epoch_losses)+1)

        plt.figure(figsize=(8,6))
        plt.plot(x_vals, self.epoch_losses,       'r-o', label="Total(Triplet + BCE)")
        plt.plot(x_vals, self.epoch_triplet_losses,'b-^', label="Triplet Loss")
        plt.plot(x_vals, self.epoch_bce_losses,    'g-s', label="BCE Loss")
        plt.title("Loss Components vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.legend()

        timestr = datetime.datetime.now().strftime("%m%d-%H%M")
        outname = os.path.join(output_dir, f"loss_components_{timestr}.png")
        plt.savefig(outname, dpi=150)
        plt.close()
        logging.info(f"Saved loss components plot: {outname}")

    def plot_multilabel_f1_curves(self, output_dir='plots'):
        import matplotlib.pyplot as plt
        import os

        if len(self.multilabel_history["macro_f1"])==0:
            logging.info("Keine multilabel_history => skip plot_multilabel_f1_curves.")
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fib_f1   = self.multilabel_history["fibrose_f1"]
        emph_f1  = self.multilabel_history["emphysem_f1"]
        nod_f1   = self.multilabel_history["nodule_f1"]
        mac_f1   = self.multilabel_history["macro_f1"]

        x_vals = range(1, len(mac_f1)+1)

        plt.figure(figsize=(8,6))
        plt.plot(x_vals, fib_f1,   'r-o', label='Fibrose F1')
        plt.plot(x_vals, emph_f1,  'g-s', label='Emphysem F1')
        plt.plot(x_vals, nod_f1,   'b-^', label='Nodule F1')
        plt.plot(x_vals, mac_f1,   'k--', label='Macro-F1', linewidth=2.0)
        plt.ylim(0,1)
        plt.title("Multi-Label F1 vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend(loc='best')

        timestr = datetime.datetime.now().strftime("%m%d-%H%M")
        aggregator_name = self.aggregator_name
        outname = os.path.join(
            output_dir,
            f"multilabel_f1_vs_epoch_{aggregator_name}_{timestr}.png"
        )
        plt.savefig(outname, dpi=150)
        plt.close()
        logging.info(f"Saved multi-label F1 curve: {outname}")