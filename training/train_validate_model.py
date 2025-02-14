import argparse
import os
import sys
import logging
import datetime
import torch
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training.trainer import TripletTrainerBase
from torch.optim.lr_scheduler import StepLR

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training + Validation (TripletTrainerBase) mit IR-Metriken. Speichert best Model."
    )

    # Pfade
    parser.add_argument("--train_csv", required=True,
                        help="Pfad zur Trainings-CSV [pid,study_yr,combination].")
    parser.add_argument("--val_csv",   required=True,
                        help="Pfad zur Validierungs-CSV [pid,study_yr,combination].")
    parser.add_argument("--data_root", required=True,
                        help="Pfad zum Verzeichnis mit den .nii.gz-Dateien.")
    parser.add_argument("--best_model_path", type=str, default="best_model.pt",
                        help="Pfad zum Speichern des besten Modells (höchstes mAP).")

    # Hyperparams
    parser.add_argument("--epochs", type=int, default=10,
                        help="Anzahl Epochen.")
    parser.add_argument("--num_triplets", type=int, default=500,
                        help="Triplets pro Epoche.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning Rate.")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Margin im TripletLoss.")

    parser.add_argument("--model_name", type=str, default="resnet18",
                        help="BaseCNN: 'resnet18' oder 'resnet50'.")
    parser.add_argument("--freeze_blocks", type=str, default=None,
                        help="z. B. '0,1' => layer1+layer2. Wenn None => kein Freeze.")

    parser.add_argument("--agg_hidden_dim", type=int, default=128,
                        help="Hidden-Dimension im Attention-MLP.")
    parser.add_argument("--agg_dropout", type=float, default=0.2,
                        help="Dropout-Rate im Attention-MLP.")

    parser.add_argument("--use_scheduler", action="store_true",
                        help="StepLR nutzen?")
    parser.add_argument("--step_size", type=int, default=10,
                        help="Scheduler step_size.")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Scheduler gamma.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Train-Device. 'cuda' oder 'cpu'.")
    parser.add_argument("--K", type=int, default=5,
                        help="K für Precision@K, Recall@K.")
    parser.add_argument("--distance_metric", type=str, default="euclidean",
                        choices=["euclidean","cosine"],
                        help="Distanzmetrik für IR-Metriken.")
    parser.add_argument("--epoch_csv", type=str, default="epoch_log.csv",
                        help="Pro-Epoch-Metriken in diese CSV schreiben.")
    parser.add_argument("--log_file", type=str, default="train_val.log",
                        help="Logfile-Pfad.")
    return parser.parse_args()

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def parse_freeze_blocks(freeze_str):
    """
    Hilfsfunktion: "0,1" -> [0,1], "None" -> None
    """
    if not freeze_str or freeze_str.lower()=="none":
        return None
    blocks = freeze_str.split(",")
    blocks = [int(x.strip()) for x in blocks]
    return blocks

def main():
    args = parse_args()
    setup_logging(args.log_file)
    logging.info("=== Starte train_validate_model.py ===")
    logging.info(f"Train CSV={args.train_csv}, Val CSV={args.val_csv}, DataRoot={args.data_root}")

    # freeze_blocks parsen
    fb = parse_freeze_blocks(args.freeze_blocks)
    logging.info(f"Freeze-Blocks: {fb}, model_name={args.model_name}")

    df_train = pd.read_csv(args.train_csv)

    trainer = TripletTrainerBase(
        df=df_train,
        data_root=args.data_root,
        device=args.device,
        lr=args.lr,
        margin=args.margin,
        model_name=args.model_name,
        freeze_blocks=fb,
        agg_hidden_dim=args.agg_hidden_dim,
        agg_dropout=args.agg_dropout
    )

    # Optional: Scheduler
    if args.use_scheduler:
        trainer.scheduler = StepLR(
            trainer.optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
        logging.info(f"Scheduler: StepLR(step_size={args.step_size}, gamma={args.gamma})")

    # Stats
    best_map = 0.0
    best_epoch = -1

    from training.triplet_sampler import TripletSampler

    df_val = pd.read_csv(args.val_csv)

    for epoch in range(1, args.epochs+1):
        logging.info(f"\n=== [TRAIN] EPOCH {epoch}/{args.epochs} ===")
        # Sampler
        sampler = TripletSampler(
            df=trainer.df,
            num_triplets=args.num_triplets,
            shuffle=True,
            top_k_negatives=3
        )
        trainer.train_one_epoch(sampler)
        sampler.reset_epoch()

        if args.use_scheduler:
            trainer.scheduler.step()
            current_lr = trainer.scheduler.get_last_lr()[0]
            logging.info(f"Nach Epoche {epoch}, LR={current_lr}")

        # === Val
        logging.info(f"=== [VAL] EPOCH {epoch}/{args.epochs} ===")
        from evaluation.metrics import compute_embeddings, compute_precision_recall_map

        emb_dict = compute_embeddings(
            trainer=trainer,
            df=df_val,
            data_root=args.data_root,
            device=args.device
        )
        val_metrics = compute_precision_recall_map(
            embeddings=emb_dict,
            K=args.K,
            distance_metric=args.distance_metric
        )
        current_map = val_metrics['mAP']
        logging.info(f"[Val-Epoch={epoch}] P@K={val_metrics['precision@K']:.4f}, "
                     f"R@K={val_metrics['recall@K']:.4f}, mAP={current_map:.4f}")

        # Check best
        if current_map>best_map:
            best_map = current_map
            best_epoch = epoch
            trainer.save_checkpoint(args.best_model_path)
            logging.info(f"=> New best mAP={best_map:.4f} @ epoch={best_epoch}, saved to {args.best_model_path}")

        if args.epoch_csv:
            _save_epoch_csv(
                epoch_csv=args.epoch_csv,
                epoch=epoch,
                trainer=trainer,
                val_metrics=val_metrics
            )

    logging.info(f"Training DONE. Best epoch={best_epoch} with Val-mAP={best_map:.4f}.")

def _save_epoch_csv(epoch_csv, epoch, trainer, val_metrics):
    """
    Hängt ans epoch_csv pro Epoche eine Zeile an:
      epoch, last_total_loss, last_trip_loss, precisionK, recallK, mAP
    """
    if len(trainer.epoch_losses)==0:
        return
    total_loss = trainer.epoch_losses[-1]
    trip_loss  = trainer.epoch_triplet_losses[-1]
    precK = val_metrics["precision@K"]
    recK  = val_metrics["recall@K"]
    mapv = val_metrics["mAP"]

    file_exists = os.path.exists(epoch_csv)
    if not file_exists:
        with open(epoch_csv, mode='w', newline='') as f:
            import csv
            writer = csv.writer(f, delimiter=';')
            header = ["epoch", "train_total_loss", "train_triplet_loss",
                      "val_precisionK", "val_recallK", "val_mAP"]
            writer.writerow(header)

    with open(epoch_csv, mode='a', newline='') as f:
        import csv
        writer = csv.writer(f, delimiter=';')
        row = [
            epoch,
            f"{total_loss:.4f}",
            f"{trip_loss:.4f}",
            f"{precK:.4f}",
            f"{recK:.4f}",
            f"{mapv:.4f}"
        ]
        writer.writerow(row)

    logging.info(f"=> Epoche {epoch} in {epoch_csv} geloggt.")


if __name__=="__main__":
    main()


# python3.11 training\train_validate_model.py `
#     --train_csv "C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\training\nlst_subset_v5_training.csv" `
#     --val_csv   "C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\validation\nlst_subset_v5_validation.csv" `
#     --data_root "D:\thesis_robert\subset_v5\NLST_subset_v5_nifti_3mm_Voxel" `
#     --epochs 30 `
#     --num_triplets 1000 `
#     --lr 1e-5 `
#     --margin 1.0 `
#     --model_name resnet18 `
#     --freeze_blocks "0,1" `
#     --agg_hidden_dim 128 `
#     --agg_dropout 0.2 `
#     --best_model_path "best_base_model.pt" `
#     --device cuda `
#     --distance_metric euclidean `
#     --K 10 `
#     --epoch_csv "epoch_metrics_base_model.csv" `
#     --log_file "train_val.log"
