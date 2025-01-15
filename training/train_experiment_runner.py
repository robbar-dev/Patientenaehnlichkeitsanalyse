#!/usr/bin/env python3
import os
import sys
import csv
import datetime
import logging
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training.trainer import TripletTrainer  # Dein Trainer
from evaluation.metrics import evaluate_model
from torch.optim.lr_scheduler import StepLR

def run_experiment(cfg):
    """
    Führt EIN Experiment aus:
      1) Instanziiert TripletTrainer mit cfg
      2) Train + Val via trainer.train_with_val(...)
      3) Gibt best Val-mAP + best_epoch zurück
    """
    # 1) DataFrames laden
    df_train = pd.read_csv(cfg["train_csv"])
    # Anmerkung: Val-CSV wird vom Trainer in train_with_val(...) genutzt,
    # dort greift er beim Val-Schritt auf evaluate_model(...) zu.

    # 2) Trainer anlegen
    trainer = TripletTrainer(
        df=df_train,
        data_root=cfg["data_root"],
        device='cuda',  # oder cfg.get("device", "cuda")
        lr=cfg["lr"],
        margin=cfg["margin"],
        roi_size=cfg["roi_size"],
        overlap=cfg["overlap"],
        pretrained=False,
        attention_hidden_dim=cfg["attention_hidden_dim"],
        dropout=cfg["dropout"],
        weight_decay=cfg["weight_decay"]
    )

    # Optional: Scheduler
    if cfg["use_scheduler"]:
        trainer.scheduler = StepLR(
            trainer.optimizer,
            step_size=cfg.get("step_size", 10),
            gamma=cfg.get("gamma", 0.5)
        )

    # 3) Train + Validate in einem Rutsch
    best_map, best_epoch = trainer.train_with_val(
        epochs=cfg["epochs"],
        num_triplets=cfg["num_triplets"],
        val_csv=cfg["val_csv"],
        data_root_val=cfg["data_root"],
        K=5,
        distance_metric="euclidean"
    )

    # => best_map, best_epoch sind das Ergebnis
    return best_map, best_epoch

def main():
    logging.basicConfig(level=logging.INFO)

    # Pfade definieren:
    TRAIN_CSV = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\training\nlst_subset_v5_training.csv"
    VAL_CSV   = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\validation\nlst_subset_v5_validation.csv"
    DATA_ROOT = r"D:\thesis_robert\NLST_subset_v5_nifti_3mm_Voxel"

    # Liste von Experiment-Konfigurationen
    experiments = [
      {
        "exp_name": "Exp01_margin1.0_lr1e-4_drop0.2",
        "train_csv": TRAIN_CSV,
        "val_csv":   VAL_CSV,
        "data_root": DATA_ROOT,

        "epochs": 10,
        "num_triplets": 500,
        "lr": 1e-4,
        "margin": 1.0,
        "roi_size": (96,96,3),
        "overlap": (10,10,1),
        "attention_hidden_dim": 128,
        "dropout": 0.2,
        "weight_decay": 1e-5,
        "use_scheduler": False
      },
      {
        "exp_name": "Exp02_margin1.2_lr1e-4_drop0.2",
        "train_csv": TRAIN_CSV,
        "val_csv":   VAL_CSV,
        "data_root": DATA_ROOT,

        "epochs": 10,
        "num_triplets": 500,
        "lr": 1e-4,
        "margin": 1.2,
        "roi_size": (96,96,3),
        "overlap": (10,10,1),
        "attention_hidden_dim": 128,
        "dropout": 0.2,
        "weight_decay": 1e-5,
        "use_scheduler": False
      },
      {
        "exp_name": "Exp03_margin0.8_lr1e-4_drop0.2",
        "train_csv": TRAIN_CSV,
        "val_csv":   VAL_CSV,
        "data_root": DATA_ROOT,

        "epochs": 10,
        "num_triplets": 500,
        "lr": 1e-4,
        "margin": 0.8,
        "roi_size": (96,96,3),
        "overlap": (10,10,1),
        "attention_hidden_dim": 128,
        "dropout": 0.2,
        "weight_decay": 1e-5,
        "use_scheduler": False
    },
    {
        "exp_name": "Exp04_margin1.0_lr5e-5_drop0.3",
        "train_csv": TRAIN_CSV,
        "val_csv":   VAL_CSV,
        "data_root": DATA_ROOT,

        "epochs": 10,
        "num_triplets": 500,
        "lr": 5e-5,
        "margin": 1.0,
        "roi_size": (96,96,3),
        "overlap": (10,10,1),
        "attention_hidden_dim": 128,
        "dropout": 0.3,
        "weight_decay": 1e-5,
        "use_scheduler": False
    },
    {
        "exp_name": "Exp05_margin1.2_lr1e-4_drop0.1",
        "train_csv": TRAIN_CSV,
        "val_csv":   VAL_CSV,
        "data_root": DATA_ROOT,

        "epochs": 10,
        "num_triplets": 500,
        "lr": 1e-4,
        "margin": 1.2,
        "roi_size": (96,96,3),
        "overlap": (10,10,1),
        "attention_hidden_dim": 128,
        "dropout": 0.1,
        "weight_decay": 1e-5,
        "use_scheduler": False
    },
    {
        "exp_name": "Exp06_margin1.5_lr1e-3_drop0.2_sched",
        "train_csv": TRAIN_CSV,
        "val_csv":   VAL_CSV,
        "data_root": DATA_ROOT,

        "epochs": 10,
        "num_triplets": 500,
        "lr": 1e-3,
        "margin": 1.5,
        "roi_size": (96,96,3),
        "overlap": (10,10,1),
        "attention_hidden_dim": 128,
        "dropout": 0.2,
        "weight_decay": 1e-5,
        "use_scheduler": True
    },
    ]

    # CSV-Ausgabedatei
    results_csv = "experiments_results.csv"

    # Header definieren
    csv_header = [
        "ExpName", "Epochs", "NumTriplets", 
        "LR", "Margin", "Dropout", "WeightDecay", 
        "UseScheduler", "BestValMAP", "BestEpoch", 
        "Timestamp"
    ]

    # Existiert Datei bereits?
    file_exists = os.path.exists(results_csv)
    if not file_exists:
        with open(results_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

    # Schleife über alle Experimente
    for cfg in experiments:
        logging.info(f"Starte Experiment: {cfg['exp_name']}")

        best_map, best_epoch = run_experiment(cfg)

        # => Speichere in CSV
        now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        row = [
            cfg["exp_name"],
            cfg["epochs"],
            cfg["num_triplets"],
            cfg["lr"],
            cfg["margin"],
            cfg["dropout"],
            cfg["weight_decay"],
            cfg["use_scheduler"],
            best_map,
            best_epoch,
            now_str
        ]
        with open(results_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        logging.info(f"Experiment {cfg['exp_name']} DONE. "
                     f"best_val_map={best_map:.4f} (epoch={best_epoch})\n")


if __name__ == "__main__":
    main()
