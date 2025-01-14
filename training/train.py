import argparse
import os
import sys
import logging
import datetime
import torch
import pandas as pd

# 1) Projektpfad
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2) Imports
from training.trainer import TripletTrainer
from evaluation.metrics import evaluate_model
from torch.optim.lr_scheduler import StepLR

def parse_args():
    parser = argparse.ArgumentParser(description="Train/Val oder Test eines TripletTrainer-Modells mit Gated-Attention-MIL + Logging/Checkpoint.")

    # Modus: train (inkl. Val) oder test
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test"],
                        help="Mode: train+val oder test (final).")

    # CSV-Pfade
    parser.add_argument("--train_csv", required=False, help="Pfad zur Trainings-CSV (nur train).")
    parser.add_argument("--val_csv",   required=False, help="Pfad zur Validierungs-CSV (nur train).")
    parser.add_argument("--test_csv",  required=False, help="Pfad zur Test-CSV (bei --mode test).")

    parser.add_argument("--data_root", required=True, help="Pfad zum Verzeichnis mit den .nii.gz-Dateien.")

    # Trainings-Hyperparameter
    parser.add_argument("--epochs", type=int, default=30, help="Anzahl Trainings-Epochen (nur train).")
    parser.add_argument("--num_triplets", type=int, default=1000, help="Anzahl Triplets pro Epoche (nur train).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Lernrate für den Adam-Optimizer.")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin im Triplet-Loss.")

    parser.add_argument("--attention_hidden_dim", type=int, default=128, help="Hidden-Dimension im Attention-MIL.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout-Rate im Attention-MIL.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight Decay für Adam.")

    # Scheduler-Optionen
    parser.add_argument("--use_scheduler", action='store_true',
                        help="Wenn gesetzt, nutzen wir StepLR.")
    parser.add_argument("--step_size", type=int, default=10,
                        help="StepSize für StepLR.")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Gamma für StepLR (Faktor zur LR-Senkung).")

    # Evaluations-Hyperparameter
    parser.add_argument("--K", type=int, default=5, help="K für Precision@K, Recall@K.")
    parser.add_argument("--distance_metric", type=str, default="euclidean",
                        choices=["euclidean", "cosine"], help="Distanzmetrik.")

    parser.add_argument("--device", type=str, default="cuda", help="Gerät, z.B. 'cuda' oder 'cpu'.")

    # Modell-Speicherpfad
    parser.add_argument("--best_model_path", type=str, default="best_model.pt",
                        help="Pfad zum Speichern/Laden des besten Modells.")

    # Logging
    parser.add_argument("--log_file", type=str, default="train.log",
                        help="Wohin das Logging geschrieben wird.")

    return parser.parse_args()

def setup_logging(log_file):
    """
    Richt das Logging ein. Schreibt ins Terminal + log_file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    args = parse_args()
    setup_logging(args.log_file)
    logging.info(f"Starte Skript in Mode={args.mode}. Logfile={args.log_file}")

    if args.mode == "train":
        # =============== TRAIN + VAL =============== #
        if not args.train_csv or not args.val_csv:
            raise ValueError("Train- und Val-CSV müssen angegeben werden, wenn --mode train.")

        logging.info(f"Train CSV={args.train_csv}, Val CSV={args.val_csv}")

        df_train = pd.read_csv(args.train_csv)

        # Trainer anlegen
        trainer = TripletTrainer(
            df=df_train,
            data_root=args.data_root,
            device=args.device,
            lr=args.lr,
            margin=args.margin,
            roi_size=(96, 96, 3),
            overlap=(10, 10, 1),
            pretrained=False,
            attention_hidden_dim=args.attention_hidden_dim,
            dropout=args.dropout,
            weight_decay=args.weight_decay
        )

        # Optional: Scheduler
        if args.use_scheduler:
            trainer.scheduler = StepLR(trainer.optimizer, step_size=args.step_size, gamma=args.gamma)
            logging.info(f"Scheduler aktiviert: StepLR(step_size={args.step_size}, gamma={args.gamma})")

        best_map = 0.0
        best_epoch = -1

        for epoch in range(1, args.epochs + 1):
            logging.info(f"\n=== EPOCH {epoch}/{args.epochs} (Train) ===")
            # 1 Epoche trainieren
            trainer.train_loop(num_epochs=1, num_triplets=args.num_triplets)
            # Scheduler Step
            if args.use_scheduler:
                trainer.scheduler.step()
                current_lr = trainer.scheduler.get_last_lr()[0]
                logging.info(f"Nach Epoche {epoch}: LR={current_lr}")

            logging.info(f"=== EPOCH {epoch}/{args.epochs} (Validation) ===")
            val_metrics = evaluate_model(
                trainer=trainer,
                data_csv=args.val_csv,
                data_root=args.data_root,
                K=args.K,
                distance_metric=args.distance_metric,
                device=args.device
            )
            current_map = val_metrics['mAP']
            logging.info(f"Val-Epoch={epoch}: mAP={current_map:.4f}, P@K={val_metrics['precision@K']:.4f}, R@K={val_metrics['recall@K']:.4f}")

            # Check if best
            if current_map > best_map:
                best_map = current_map
                best_epoch = epoch
                # Zeitstempel
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                checkpoint_name = f"best_model_{timestamp}.pt"

                # Speichere Checkpoint
                trainer.save_checkpoint(checkpoint_name)
                logging.info(f"=> Neuer Best mAP={best_map:.4f} in Epoche {best_epoch}. "
                             f"Checkpoint saved as {checkpoint_name}")

        logging.info(f"Training DONE. Best epoch war {best_epoch} mit Val-mAP={best_map:.4f}.")

    elif args.mode == "test":
        # =============== TEST =============== #
        if not args.test_csv:
            raise ValueError("Test-CSV muss angegeben werden, wenn --mode test.")
        if not os.path.exists(args.best_model_path):
            raise FileNotFoundError(f"best_model_path ({args.best_model_path}) existiert nicht.")

        logging.info(f"Starte TEST: Test-CSV={args.test_csv}, best_model={args.best_model_path}")

        # Erstelle Trainer (leer, wir laden State)
        trainer = TripletTrainer(
            df=None,
            data_root=args.data_root,
            device=args.device,
            lr=args.lr,
            margin=args.margin,
            roi_size=(96, 96, 3),
            overlap=(10, 10, 1),
            pretrained=False,
            attention_hidden_dim=args.attention_hidden_dim,
            dropout=args.dropout,
            weight_decay=args.weight_decay
        )

        # Lade bestes Modell
        trainer.load_checkpoint(args.best_model_path)

        # Evaluate Test
        test_metrics = evaluate_model(
            trainer=trainer,
            data_csv=args.test_csv,
            data_root=args.data_root,
            K=args.K,
            distance_metric=args.distance_metric,
            device=args.device
        )
        logging.info(f"TEST DONE: P@K={test_metrics['precision@K']:.4f}, R@K={test_metrics['recall@K']:.4f}, mAP={test_metrics['mAP']:.4f}")

    else:
        raise ValueError("Unbekannter Mode. Verwende --mode train oder --mode test.")

if __name__ == "__main__":
    main()

# TRAIN & EVALUATE

# python3.11 training\train.py `
#     --mode train `
#     --train_csv "C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\training\nlst_subset_v5_training.csv" `
#     --val_csv "C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\validation\nlst_subset_v5_validation.csv" `
#     --data_root "D:\thesis_robert\NLST_subset_v5_nifti_3mm_Voxel" `
#     --epochs 15 `
#     --num_triplets 500 `
#     --lr 1e-4 `
#     --margin 1.0 `
#     --best_model_path "best_model.pt"


# TEST

# python3.11 training\train.py `
#     --mode test `
#     --test_csv  "C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\test\nlst_subset_v5_test.csv" `
#     --data_root "D:\thesis_robert\NLST_subset_v5_nifti_3mm_Voxel" `
#     --best_model_path "best_model.pt" `
#     --K 5
