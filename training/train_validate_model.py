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

from training.trainer import TripletTrainer
from torch.optim.lr_scheduler import StepLR

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training + Validation (TripletTrainer) mit IR-Metriken und Multi-Label + Hard Negatives."
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

    # Training-Stufen
    parser.add_argument("--epochs_stage1", type=int, default=5,
                        help="Anzahl Epochen Stage1 (normal Sampler).")
    parser.add_argument("--epochs_stage2", type=int, default=15,
                        help="Anzahl Epochen Stage2 (Hard Negatives).")
    parser.add_argument("--num_triplets", type=int, default=500,
                        help="Triplets pro Epoche.")
    
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning Rate.")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Margin im TripletLoss.")

    # CNN / Aggregator
    parser.add_argument("--model_name", type=str, default="resnet18",
                        help="BaseCNN: 'resnet18' oder 'resnet50'.")
    parser.add_argument("--freeze_blocks", type=str, default=None,
                        help="z. B. '0,1' => layer1+layer2. Wenn None => kein Freeze.")
    parser.add_argument("--aggregator_name", type=str, default="mil",
                        choices=["mil","max","mean"],
                        help="Aggregator: MIL / Max / Mean")

    parser.add_argument("--agg_hidden_dim", type=int, default=128,
                        help="Hidden-Dimension im Attention-MLP (wenn aggregator=mil).")
    parser.add_argument("--agg_dropout", type=float, default=0.2,
                        help="Dropout-Rate im Attention-MLP (wenn aggregator=mil).")
    
    parser.add_argument("--device", type=str, default="cuda",
                        help="Train-Device. 'cuda' oder 'cpu'.")

    # Hyperparams IR
    parser.add_argument("--K", type=int, default=5,
                        help="K für Precision@K, Recall@K.")
    parser.add_argument("--distance_metric", type=str, default="euclidean",
                        choices=["euclidean","cosine"],
                        help="Distanzmetrik für IR-Metriken.")

    # Schalter: Scheduler
    parser.add_argument("--use_scheduler", action="store_true",
                        help="StepLR nutzen?")
    parser.add_argument("--step_size", type=int, default=10,
                        help="Scheduler step_size.")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Scheduler gamma.")

    # CSV + Logging
    parser.add_argument("--epoch_csv", type=str, default="epoch_log.csv",
                        help="Pro-Epoch-Metriken in diese CSV schreiben.")
    parser.add_argument("--log_file", type=str, default="train_val.log",
                        help="Logfile-Pfad.")

    # Augmentation
    parser.add_argument("--do_augmentation", action="store_true",
                        help="Wenn gesetzt, wird random Augmentation im SinglePatientDataset aktiviert.")

    # Multi-Label
    parser.add_argument("--lambda_bce", type=float, default=1.0,
                        help="Gewicht für BCE im Triplet+BCE Mix.")

    # 2-Stage Training vs. 1-Stage
    parser.add_argument("--two_stage", action="store_true",
                        help="Wenn gesetzt, wird train_with_val_2stage ausgeführt, sonst normal train_with_val.")

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
    Hilfsfunktion: "0,1" => [0,1], "None" => None
    """
    if not freeze_str or freeze_str.lower()=="none":
        return None
    blocks = freeze_str.split(",")
    blocks = [int(x.strip()) for x in blocks]
    return blocks

def main():
    args = parse_args()
    setup_logging(args.log_file)
    logging.info("=== Starte train_validate_model.py (Multi-Label + HardNeg) ===")
    logging.info(f"Train CSV={args.train_csv}, Val CSV={args.val_csv}, DataRoot={args.data_root}")

    fb = parse_freeze_blocks(args.freeze_blocks)
    logging.info(f"Freeze-Blocks: {fb}, model_name={args.model_name}, aggregator={args.aggregator_name}")

    # => DataFrame laden
    df_train = pd.read_csv(args.train_csv)

    trainer = TripletTrainer(
        df=df_train,
        data_root=args.data_root,
        device=args.device,
        lr=args.lr,
        margin=args.margin,
        model_name=args.model_name,
        freeze_blocks=fb,
        aggregator_name=args.aggregator_name,
        agg_hidden_dim=args.agg_hidden_dim,
        agg_dropout=args.agg_dropout,
        do_augmentation=args.do_augmentation,
        lambda_bce=args.lambda_bce
    )

    # => Scheduler optional
    if args.use_scheduler:
        trainer.scheduler = StepLR(
            trainer.optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
        logging.info(f"Scheduler: StepLR(step_size={args.step_size}, gamma={args.gamma})")

    logging.info(f"Will run two_stage={args.two_stage}. " 
                 f"Stage1={args.epochs_stage1}, Stage2={args.epochs_stage2}, total={args.epochs_stage1+args.epochs_stage2} epochs.")

    if args.two_stage:
        # 2-Stage => Hard Negatives
        best_map, best_epoch = trainer.train_with_val_2stage(
            epochs_stage1=args.epochs_stage1,
            epochs_stage2=args.epochs_stage2,
            num_triplets=args.num_triplets,
            val_csv=args.val_csv,
            data_root_val=args.data_root,
            K=args.K,
            distance_metric=args.distance_metric,
            visualize_every=5,
            visualize_method='tsne',
            output_dir='plots',
            epoch_csv_path=args.epoch_csv
        )
    else:
        # 1-Stage normal
        best_map, best_epoch = trainer.train_with_val(
            epochs=args.epochs_stage1,    # ggf. hier nur 1 Wert
            num_triplets=args.num_triplets,
            val_csv=args.val_csv,
            data_root_val=args.data_root,
            K=args.K,
            distance_metric=args.distance_metric,
            visualize_every=5,
            visualize_method='tsne',
            output_dir='plots',
            epoch_csv_path=args.epoch_csv
        )

    logging.info(f"Training DONE. best_val_map={best_map:.4f} at epoch={best_epoch}.")

if __name__=="__main__":
    main()

#-------------ohne Hard Mining----------------
# python3.11 training\train_validate_model.py `
#     --train_csv "C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\training\nlst_subset_v5_training.csv" `
#     --val_csv   "C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\training\nlst_subset_v5_training.csv" `
#     --data_root "D:\thesis_robert\subset_v5\NLST_subset_v5_nifti_3mm_Voxel" `
#     --epochs_stage1 30 `
#     --num_triplets 1000 `
#     --lr 1e-5 `
#     --margin 1.0 `
#     --model_name resnet18 `
#     --freeze_blocks "0" `
#     --agg_hidden_dim 128 `
#     --agg_dropout 0.2 `
#     --best_model_path "best_base_model.pt" `
#     --device cuda `
#     --distance_metric euclidean `
#     --K 10 `
#     --epoch_csv "epoch_metrics_base_model.csv" `
#     --log_file "train_val.log"

#-------------mit Hard Mining ohne Augmentierung ----------------
# python3.11 training\train_validate_model.py `
#     --train_csv "C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\training\nlst_subset_v5_training.csv" `
#     --val_csv   "C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\training\nlst_subset_v5_training.csv" `
#     --data_root "D:\thesis_robert\subset_v5\NLST_subset_v5_nifti_3mm_Voxel" `
#     --epochs_stage1 5 `
#     --epochs_stage2 25 `
#     --two_stage `
#     --num_triplets 1 `
#     --lr 1e-5 `
#     --margin 1.0 `
#     --model_name resnet18 `
#     --freeze_blocks "0" `
#     --agg_hidden_dim 128 `
#     --agg_dropout 0.2 `
#     --best_model_path "best_base_model.pt" `
#     --device cuda `
#     --distance_metric euclidean `
#     --K 10 `
#     --epoch_csv "epoch_metrics_base_model.csv" `
#     --log_file "train_val.log"

#-------------mit Hard Mining ohne Augmentierung auf Validierungsdate ----------------
# python3.11 training\train_validate_model.py `
#     --train_csv "C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\training\nlst_subset_v5_training.csv" `
#     --val_csv   "C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\validation\nlst_subset_v5_validation.csv" `
#     --data_root "D:\thesis_robert\subset_v5\NLST_subset_v5_nifti_3mm_Voxel" `
#     --epochs_stage1 5 `
#     --epochs_stage2 25 `
#     --two_stage `
#     --num_triplets 1 `
#     --lr 1e-5 `
#     --margin 1.0 `
#     --model_name resnet18 `
#     --freeze_blocks "0" `
#     --agg_hidden_dim 128 `
#     --agg_dropout 0.2 `
#     --best_model_path "best_base_model.pt" `
#     --device cuda `
#     --distance_metric euclidean `
#     --K 10 `
#     --epoch_csv "epoch_metrics_base_model.csv" `
#     --log_file "train_val.log"

#-------------mit Hard Mining mit Augmentierung ----------------
# python3.11 training\train_validate_model.py `
#     --train_csv "C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\training\nlst_subset_v5_training.csv" `
#     --val_csv   "C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\training\nlst_subset_v5_training.csv" `
#     --data_root "D:\thesis_robert\subset_v5\NLST_subset_v5_nifti_3mm_Voxel" `
#     --epochs_stage1 5 `
#     --epochs_stage2 25 `
#     --two_stage `
#     --num_triplets 1 `
#     --lr 1e-5 `
#     --margin 1.0 `
#     --model_name resnet18 `
#     --freeze_blocks "0" `
#     --agg_hidden_dim 128 `
#     --agg_dropout 0.2 `
#     --do_augmentation `
#     --best_model_path "best_base_model.pt" `
#     --device cuda `
#     --distance_metric euclidean `
#     --K 10 `
#     --epoch_csv "epoch_metrics_base_model.csv" `
#     --log_file "train_val.log"
