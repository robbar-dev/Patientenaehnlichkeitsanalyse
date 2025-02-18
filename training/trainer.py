import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import datetime
import csv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from training.triplet_sampler import BinaryTripletSampler
from training.triplet_sampler_hard_negative import HardNegativeBinaryTripletSampler

from evaluation.metrics import compute_embeddings, compute_precision_recall_map

from evaluation.binary_metrics import evaluate_binary_classification

def parse_combo_str_to_label_binary(combo_str):
    """
    '1-0-0' => 1 (Abnormal)
    '0-0-1' => 0 (Normal)
    """
    if combo_str.startswith('1-0-0'):
        return 1
    else:
        return 0


class TripletTrainer(nn.Module):
    """
    Trainer für Binary-Klassifikation,
    plus TripletMarginLoss (Hard Negative Mining 2-stage),
    plus IR-Metriken (Precision@K, Recall@K, mAP),
    plus binäre Metriken (ACC, AUC, ConfusionMatrix, ROC).
    
    Umbau gegenüber Multi-Label-Version:
     - classifier = Linear(512, 1)
     - BCE => shape(1,1) pro Patient
     - evaluate_binary_classification(...) statt evaluate_multilabel_classification
     - binclass_history für ACC, AUC
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
        lambda_bce=1.0
    ):
        super().__init__()
        self.df = df.copy()  
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

        from model.base_cnn import BaseCNN
        self.base_cnn = BaseCNN(
            model_name=self.model_name,
            pretrained=True,
            freeze_blocks=self.freeze_blocks
        ).to(device)

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
            raise ValueError(f"Unbekannte aggregator_name={aggregator_name} - nutze 'mil','max' oder 'mean'.")

        self.triplet_loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2)

        
        # BINARY-Kopf => 1 Dimension
        self.classifier = nn.Linear(512, 1).to(device)
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

        params = list(self.base_cnn.parameters()) \
               + list(self.mil_agg.parameters()) \
               + list(self.classifier.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr)

        self.epoch_losses = []         
        self.epoch_triplet_losses = []  
        self.epoch_bce_losses = []      

        # IR (Precision@K, Recall@K, mAP)
        self.metric_history = {
            "precision": [],
            "recall": [],
            "mAP": []
        }

        # Binäre Metriken
        self.binclass_history = {
            "acc": [],
            "auc": []
        }

        self.best_val_map = 0.0
        self.best_val_epoch = -1

        logging.info("[TripletTrainer] Initialized (Binary Version)")
        logging.info(f"lr={lr}, margin={margin}, aggregator={aggregator_name}, "
                     f"freeze_blocks={freeze_blocks}, agg_hidden_dim={agg_hidden_dim}, "
                     f"agg_dropout={agg_dropout}, do_augmentation={do_augmentation}, "
                     f"lambda_bce={lambda_bce}")

        self.df['multi_label'] = self.df['combination'].apply(
            lambda c: [parse_combo_str_to_label_binary(c)]
        )

    def _forward_patient(self, pid, study_yr, do_train=True):
        """
        => Patches => base_cnn => aggregator => (1,512) => classifier => (1,1)
        => returns (embedding, logits)
        """
        from training.data_loader import SinglePatientDataset
        ds = SinglePatientDataset(
            data_root=self.data_root,
            pid=pid,
            study_yr=study_yr,
            roi_size=self.roi_size,
            overlap=self.overlap,
            skip_factor=1,
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

        if len(patch_embs) == 0:
            dummy = torch.zeros((1,512), device=self.device, requires_grad=do_train)
            logits= self.classifier(dummy)  # => (1,1)
            return dummy, logits

        patch_embs = torch.cat(patch_embs, dim=0)  # => (N,512)
        patient_emb = self.mil_agg(patch_embs)      # => (1,512)
        logits = self.classifier(patient_emb)       # => (1,1)
        return patient_emb, logits

    def compute_patient_embedding(self, pid, study_yr):
        """
        => Für IR => (1,512) => no augmentation
        """
        emb, logits = self._forward_patient(pid, study_yr, do_train=False)
        return emb  # => (1,512)

    def train_one_epoch(self, sampler):
        total_loss = 0.0
        total_trip = 0.0
        total_bce  = 0.0
        steps = 0

        for step,(a_info, p_info, n_info) in enumerate(sampler):
            a_pid,a_sy = a_info['pid'], a_info['study_yr']
            p_pid,p_sy = p_info['pid'], p_info['study_yr']
            n_pid,n_sy = n_info['pid'], n_info['study_yr']

            # multi_label => [0] or [1]
            a_lab = a_info['multi_label']  # => e.g. [1]
            p_lab = p_info['multi_label']
            n_lab = n_info['multi_label']

            a_emb, a_logits = self._forward_patient(a_pid,a_sy,do_train=True)
            p_emb, p_logits = self._forward_patient(p_pid,p_sy,do_train=True)
            n_emb, n_logits = self._forward_patient(n_pid,n_sy,do_train=True)

            trip_loss = self.triplet_loss_fn(a_emb, p_emb, n_emb)

            # BCE => shape(1,1)
            a_lab_t = torch.tensor(a_lab, dtype=torch.float32, device=self.device).unsqueeze(0) # => (1,1)
            p_lab_t = torch.tensor(p_lab, dtype=torch.float32, device=self.device).unsqueeze(0)
            n_lab_t = torch.tensor(n_lab, dtype=torch.float32, device=self.device).unsqueeze(0)

            bce_a = self.bce_loss_fn(a_logits, a_lab_t)
            bce_p = self.bce_loss_fn(p_logits, p_lab_t)
            bce_n = self.bce_loss_fn(n_logits, n_lab_t)
            bce_loss = (bce_a + bce_p + bce_n)/3.0

            loss = trip_loss + self.lambda_bce*bce_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_trip += trip_loss.item()
            total_bce  += bce_loss.item()
            steps+=1

            if step%250==0:
                logging.info(f"[Step {step}] total={loss.item():.4f}, trip={trip_loss.item():.4f}, BCE={bce_loss.item():.4f}")

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

            emb = self.compute_patient_embedding(pid, study_yr)
            emb_np = emb.squeeze(0).detach().cpu().numpy()
            embeddings_list.append(emb_np)
            combos.append(combo)

        embeddings_arr = np.array(embeddings_list)

        if method.lower()=='tsne':
            projector = TSNE(n_components=2, random_state=42)
        else:
            logging.warning("Nur t-SNE implementiert, fallback = t-SNE")
            projector = TSNE(n_components=2, random_state=42)

        coords_2d = projector.fit_transform(embeddings_arr)

        plt.figure(figsize=(8,6))
        unique_combos = sorted(list(set(combos)))
        for c in unique_combos:
            idxs = [ix for ix,val in enumerate(combos) if val==c]
            plt.scatter(coords_2d[idxs,0], coords_2d[idxs,1], label=str(c), alpha=0.6)

        plt.title(f"t-SNE Embeddings - EPOCH={epoch}")
        plt.legend(loc='best')

        aggregator_name = getattr(self.mil_agg,'__class__').__name__
        timestr = datetime.datetime.now().strftime("%m%d-%H%M")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fname = os.path.join(output_dir,f"Emb_{method}_{aggregator_name}_epoch{epoch:03d}_{timestr}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        logging.info(f"Saved embedding plot => {fname}")

    def _train_one_epoch_internal(self, sampler):
        total_loss = 0.0
        total_trip = 0.0
        total_bce  = 0.0
        steps = 0

        for step, (a_info, p_info, n_info) in enumerate(sampler):
            a_pid, a_sy = a_info['pid'], a_info['study_yr']
            p_pid, p_sy = p_info['pid'], p_info['study_yr']
            n_pid, n_sy = n_info['pid'], n_info['study_yr']

            a_lab = a_info['multi_label']
            p_lab = p_info['multi_label']
            n_lab = n_info['multi_label']

            a_emb,a_logits = self._forward_patient(a_pid,a_sy,do_train=True)
            p_emb,p_logits = self._forward_patient(p_pid,p_sy,do_train=True)
            n_emb,n_logits = self._forward_patient(n_pid,n_sy,do_train=True)

            trip_loss= self.triplet_loss_fn(a_emb,p_emb,n_emb)

            a_lab_t = torch.tensor(a_lab, dtype=torch.float32, device=self.device).unsqueeze(0)
            p_lab_t = torch.tensor(p_lab, dtype=torch.float32, device=self.device).unsqueeze(0)
            n_lab_t = torch.tensor(n_lab, dtype=torch.float32, device=self.device).unsqueeze(0)

            bce_a = self.bce_loss_fn(a_logits,a_lab_t)
            bce_p = self.bce_loss_fn(p_logits,p_lab_t)
            bce_n = self.bce_loss_fn(n_logits,n_lab_t)
            bce_loss = (bce_a+bce_p+bce_n)/3.0

            loss= trip_loss + self.lambda_bce*bce_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_trip += trip_loss.item()
            total_bce  += bce_loss.item()
            steps+=1

            if step%250==0:
                logging.info(f"[Step={step}] total={loss.item():.4f}, trip={trip_loss.item():.4f}, BCE={bce_loss.item():.4f}")

        if steps>0:
            avg_loss= total_loss/steps
            avg_trip= total_trip/steps
            avg_bce = total_bce/steps
        else:
            avg_loss=0; avg_trip=0; avg_bce=0

        self.epoch_losses.append(avg_loss)
        self.epoch_triplet_losses.append(avg_trip)
        self.epoch_bce_losses.append(avg_bce)
        logging.info(f"=> EPOCH Loss={avg_loss:.4f}, Trip={avg_trip:.4f}, BCE={avg_bce:.4f}")

    def train_loop(self, num_epochs=5, num_triplets=100):
        sampler = BinaryTripletSampler(self.df, num_triplets, shuffle=True)

        for epoch in range(1,num_epochs+1):
            logging.info(f"=== EPOCH {epoch}/{num_epochs} ===")
            self.train_one_epoch(sampler)
            sampler.reset_epoch()

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
        output_dir='plots',
        epoch_csv_path=None
    ):
        """
        am Ende jeder Epoche:
          - IR (Precision,Recall,mAP)
          - Binary Evaluate => ACC, AUC + optional ConfMatrix/ROC
          - CSV-Log + Plots
          - Checkpoint => Best IR-mAP
        """
        df_val = pd.read_csv(val_csv)

        # Reset
        self.epoch_losses=[]
        self.epoch_triplet_losses=[]
        self.epoch_bce_losses=[]
        self.metric_history= {"precision":[],"recall":[],"mAP":[]}
        self.binclass_history= {"acc":[],"auc":[]}
        self.best_val_map=0.0
        self.best_val_epoch=-1

        if epoch_csv_path and not os.path.exists(epoch_csv_path):
            with open(epoch_csv_path, 'w', newline='') as f:
                w = csv.writer(f, delimiter=';')
                header=["Epoch","TripLoss","BCE","Precision@K","Recall@K","mAP","ACC","AUC"]
                w.writerow(header)

        sampler = BinaryTripletSampler(self.df,num_triplets,shuffle=True)

        for epoch in range(1,epochs+1):
            logging.info(f"=== EPOCH {epoch}/{epochs} ===")
            self.train_one_epoch(sampler)
            sampler.reset_epoch()

            # IR 
            emb_dict= compute_embeddings(self, df_val, data_root_val, device=self.device)
            val_metrics= compute_precision_recall_map(emb_dict,K=K,distance_metric=distance_metric)
            precK= val_metrics["precision@K"]
            recK = val_metrics["recall@K"]
            mapv = val_metrics["mAP"]
            self.metric_history["precision"].append(precK)
            self.metric_history["recall"].append(recK)
            self.metric_history["mAP"].append(mapv)

            logging.info(f"[Val-Epoch={epoch}] P@K={precK:.4f}, R@K={recK:.4f}, mAP={mapv:.4f}")
            if mapv>self.best_val_map:
                self.best_val_map= mapv
                self.best_val_epoch= epoch
                logging.info(f"=> New best mAP={mapv:.4f} @ epoch={epoch}")
                self.save_checkpoint("best_model.pt")

            # Binary Evaluate
            bin_res = evaluate_binary_classification(
                trainer=self,
                df=df_val,
                data_root=data_root_val,
                device=self.device,
                threshold=0.5,
                do_plot_roc=False,
                do_plot_cm=False
            )
            acc_val= bin_res["acc"]
            auc_val= bin_res["auc"]
            logging.info(f"Binary => ACC={acc_val:.4f}, AUC={auc_val:.4f}")
            self.binclass_history["acc"].append(acc_val)
            self.binclass_history["auc"].append(auc_val)

            # t-SNE
            if epoch%visualize_every==0:
                self.visualize_embeddings(
                    df=df_val,
                    data_root=data_root_val,
                    method=visualize_method,
                    epoch=epoch,
                    output_dir=output_dir
                )

            if epoch_csv_path:
                self._write_epoch_csv_binary(epoch, epoch_csv_path,
                                             precK, recK, mapv,
                                             acc_val, auc_val)

        self.plot_loss_components(output_dir=output_dir)
        self.plot_metric_curves(output_dir=output_dir)
        self.plot_acc_auc_curves(output_dir=output_dir) 

        return self.best_val_map,self.best_val_epoch

    def _write_epoch_csv_binary(self, epoch, csv_path,
                                precK, recK, mapv,
                                acc_val, auc_val):
        if len(self.epoch_triplet_losses)==0 or len(self.epoch_bce_losses)==0:
            return
        trip = self.epoch_triplet_losses[-1]
        bce  = self.epoch_bce_losses[-1]

        with open(csv_path,'a',newline='') as f:
            w = csv.writer(f, delimiter=';')
            row=[epoch,f"{trip:.4f}",f"{bce:.4f}",
                 f"{precK:.4f}",f"{recK:.4f}",f"{mapv:.4f}",
                 f"{acc_val:.4f}",f"{auc_val:.4f}"]
            w.writerow(row)
        logging.info(f"=> Epoche {epoch} logged in {csv_path}")

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
        """
        2-Stage: Stage1 -> normal Sampler, Stage2 -> Hard Negative
        Evaluate -> IR + Binary -> CSV -> Checkpoint
        """
        df_val = pd.read_csv(val_csv)

        # Reset
        self.epoch_losses=[]
        self.epoch_triplet_losses=[]
        self.epoch_bce_losses=[]
        self.metric_history={"precision":[],"recall":[],"mAP":[]}
        self.binclass_history={"acc":[],"auc":[]}
        self.best_val_map=0.0
        self.best_val_epoch=-1

        if epoch_csv_path and not os.path.exists(epoch_csv_path):
            with open(epoch_csv_path,'w',newline='') as f:
                w=csv.writer(f, delimiter=';')
                head=["Stage","Epoch","TripLoss","BCELoss","P@K","R@K","mAP","ACC","AUC"]
                w.writerow(head)

        total_epochs= epochs_stage1+ epochs_stage2
        current_epoch=0

        sampler_normal = BinaryTripletSampler(self.df,num_triplets,shuffle=True)

        # Stage1
        for e in range(1,epochs_stage1+1):
            current_epoch+=1
            stage="Stage1"
            logging.info(f"=== {stage}-Epoch {current_epoch}/{total_epochs} ===")

            self._train_one_epoch_internal(sampler_normal)
            sampler_normal.reset_epoch()

            self._eval_and_visualize_binary(
                stage, current_epoch, df_val, val_csv,
                data_root_val, K, distance_metric,
                visualize_every, visualize_method,
                output_dir, epoch_csv_path
            )

        # Stage2 -> Hard Negative
        for e in range(1,epochs_stage2+1):
            current_epoch+=1
            stage="Stage2"
            logging.info(f"=== {stage}-Epoch {current_epoch}/{total_epochs} ===")

            if (e%5)==1:
                sampler_hard= HardNegativeBinaryTripletSampler(
                    df=self.df,
                    trainer=self,
                    num_triplets=num_triplets,
                    device=self.device
                )
            self._train_one_epoch_internal(sampler_hard)

            self._eval_and_visualize_binary(
                stage, current_epoch, df_val, val_csv,
                data_root_val, K, distance_metric,
                visualize_every, visualize_method,
                output_dir, epoch_csv_path
            )

        # Plots
        self.plot_loss_components(output_dir=output_dir)
        self.plot_metric_curves(output_dir=output_dir)
        self.plot_acc_auc_curves(output_dir=output_dir)

        return self.best_val_map,self.best_val_epoch

    def _eval_and_visualize_binary(
        self,
        stage,
        epoch,
        df_val,
        val_csv,
        data_root_val,
        K,
        distance_metric,
        visualize_every,
        visualize_method,
        output_dir,
        epoch_csv_path
    ):
        # IR
        emb_dict= compute_embeddings(self,df_val,data_root_val,device=self.device)
        val_m= compute_precision_recall_map(emb_dict,K=K,distance_metric=distance_metric)
        precK= val_m["precision@K"]
        recK = val_m["recall@K"]
        mapv = val_m["mAP"]
        self.metric_history["precision"].append(precK)
        self.metric_history["recall"].append(recK)
        self.metric_history["mAP"].append(mapv)

        if mapv>self.best_val_map:
            self.best_val_map= mapv
            self.best_val_epoch= epoch
            logging.info(f"=> New best mAP={mapv:.4f} at epoch={epoch}")
            self.save_checkpoint("best_model.pt")

        logging.info(f"[Val-Ep={epoch}] P@K={precK:.4f}, R@K={recK:.4f}, mAP={mapv:.4f}")

        # Binär
        bin_res= evaluate_binary_classification(
            trainer=self,
            df=df_val,
            data_root=data_root_val,
            device=self.device,
            threshold=0.5
        )
        acc= bin_res["acc"]
        auc_val= bin_res["auc"]
        self.binclass_history["acc"].append(acc)
        self.binclass_history["auc"].append(auc_val)
        logging.info(f"Binary => ACC={acc:.4f}, AUC={auc_val:.4f}")

        # t-SNE
        if epoch%visualize_every==0:
            self.visualize_embeddings(df_val, data_root_val, visualize_method, epoch, output_dir)

        # CSV
        if epoch_csv_path:
            self._write_csv_2stage_binary(stage,epoch,epoch_csv_path,precK,recK,mapv,acc,auc_val)

    def _write_csv_2stage_binary(self, stage, epoch, csv_path, precK, recK, mapv, acc, auc_val):
        if len(self.epoch_triplet_losses)==0 or len(self.epoch_bce_losses)==0:
            return
        trip= self.epoch_triplet_losses[-1]
        bce = self.epoch_bce_losses[-1]

        with open(csv_path,'a',newline='') as f:
            w= csv.writer(f, delimiter=';')
            row = [
                stage, epoch,
                f"{trip:.4f}",
                f"{bce:.4f}",
                f"{precK:.4f}",
                f"{recK:.4f}",
                f"{mapv:.4f}",
                f"{acc:.4f}",
                f"{auc_val:.4f}"
            ]
            w.writerow(row)
        logging.info(f"=> Epoche {epoch} (stage={stage}) logged => {csv_path}")

    # Plot: ACC,AUC vs. Epoche
    def plot_acc_auc_curves(self, output_dir='plots'):
        import matplotlib.pyplot as plt
        import os
        if len(self.binclass_history["acc"])==0:
            logging.info("Keine binclass_history => skip plot_acc_auc_curves.")
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        x_vals= range(1,len(self.binclass_history["acc"])+1)
        accs= self.binclass_history["acc"]
        aucs= self.binclass_history["auc"]

        plt.figure(figsize=(7,5))
        plt.plot(x_vals, accs, 'b-o', label='ACC')
        plt.plot(x_vals, aucs, 'r-^', label='AUC')
        plt.ylim(0,1)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Binary Classification: ACC & AUC vs. Epoch")
        plt.legend(loc='best')
        timestr = datetime.datetime.now().strftime("%m%d-%H%M")
        outname = os.path.join(output_dir,f"acc_auc_vs_epoch_{timestr}.png")
        plt.savefig(outname, dpi=150)
        plt.close()
        logging.info(f"Saved ACC/AUC curves => {outname}")

    def plot_loss_components(self, output_dir='plots'):
        import os
        import matplotlib.pyplot as plt
        if len(self.epoch_losses)==0:
            logging.info("Keine Verlaufsdaten => skip plot_loss_components.")
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        x_vals= range(1,len(self.epoch_losses)+1)

        plt.figure(figsize=(8,6))
        plt.plot(x_vals, self.epoch_losses,       'r-o', label="Total(Trip+BCE)")
        plt.plot(x_vals, self.epoch_triplet_losses,'b-^', label="Triplet")
        plt.plot(x_vals, self.epoch_bce_losses,    'g-s', label="BCE")
        plt.title("Loss Components vs. Epoch (Binary)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.legend()

        timestr= datetime.datetime.now().strftime("%m%d-%H%M")
        outname= os.path.join(output_dir, f"loss_components_{timestr}.png")
        plt.savefig(outname,dpi=150)
        plt.close()
        logging.info(f"Saved loss components => {outname}")

    def plot_metric_curves(self, output_dir='plots'):
        import os
        import matplotlib.pyplot as plt
        if len(self.metric_history["mAP"])==0:
            logging.info("Keine IR-Metriken => skip plot_metric_curves.")
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        x_vals= range(1,len(self.metric_history["mAP"])+1)
        prec= self.metric_history["precision"]
        rec= self.metric_history["recall"]
        maps= self.metric_history["mAP"]

        plt.figure(figsize=(8,6))
        plt.plot(x_vals, prec, 'g-o', label='Precision@K')
        plt.plot(x_vals, rec,  'm-s', label='Recall@K')
        plt.plot(x_vals, maps, 'r-^', label='mAP')
        plt.ylim(0,1)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Precision@K, Recall@K, mAP vs. Epoch")
        plt.legend()

        timestr= datetime.datetime.now().strftime("%m%d-%H%M")
        outname= os.path.join(output_dir,f"ir_metrics_vs_epoch_{timestr}.png")
        plt.savefig(outname,dpi=150)
        plt.close()
        logging.info(f"Saved IR metric curves => {outname}")

    def save_checkpoint(self, path):
        ckpt={
            'base_cnn': self.base_cnn.state_dict(),
            'mil_agg': self.mil_agg.state_dict(),
            'classifier': self.classifier.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(ckpt,path)
        logging.info(f"Checkpoint saved => {path}")

    def load_checkpoint(self, path):
        ckpt= torch.load(path, map_location=self.device)
        self.base_cnn.load_state_dict(ckpt['base_cnn'])
        self.mil_agg.load_state_dict(ckpt['mil_agg'])
        self.classifier.load_state_dict(ckpt['classifier'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        logging.info(f"Checkpoint loaded from {path}")
