import argparse
import os
import sys
import logging
import torch
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training.trainer import TripletTrainerBase
from evaluation.metrics import compute_embeddings, compute_precision_recall_map

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test eines gespeicherten BaseModels via IR-Metriken (Precision@K, Recall@K, mAP)."
    )

    parser.add_argument("--test_csv", required=True,
                        help="Pfad zur Test-CSV [pid, study_yr, combination].")
    parser.add_argument("--data_root", required=True,
                        help="Pfad zu den .nii.gz-Dateien.")
    parser.add_argument("--model_path", type=str, default="best_model.pt",
                        help="Pfad zum gespeicherten Checkpoint.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Gerät, z.B. 'cuda' oder 'cpu'.")
    parser.add_argument("--K", type=int, default=5,
                        help="K für Precision@K, Recall@K.")
    parser.add_argument("--distance_metric", type=str, default="euclidean",
                        choices=["euclidean","cosine"],
                        help="Distanzmetrik.")
    parser.add_argument("--log_file", type=str, default="test.log",
                        help="Wohin das Logging geschrieben wird.")
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

def main():
    args = parse_args()
    setup_logging(args.log_file)

    logging.info("=== TEST / INFERENCE ===")
    logging.info(f"Test-CSV={args.test_csv}, data_root={args.data_root}, model_path={args.model_path}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Checkpoint {args.model_path} existiert nicht.")

    df_test = pd.read_csv(args.test_csv)

    # Wir erstellen einen leeren TrainerBase (ohne df), laden dann die Weights
    trainer = TripletTrainerBase(
        df=None,  # None, da nur Inference 
        data_root=args.data_root,
        device=args.device,
        lr=1e-4,
        margin=1.0
    )

    # Lade Checkpoint
    trainer.load_checkpoint(args.model_path)

    # => Evaluate
    emb_dict = compute_embeddings(
        trainer=trainer,
        df=df_test,
        data_root=args.data_root,
        device=args.device
    )
    test_metrics = compute_precision_recall_map(
        embeddings=emb_dict,
        K=args.K,
        distance_metric=args.distance_metric
    )

    logging.info(f"TEST => Precision@K={test_metrics['precision@K']:.4f}, "
                 f"Recall@K={test_metrics['recall@K']:.4f}, mAP={test_metrics['mAP']:.4f}")

if __name__=="__main__":
    main()

# python test_model.py \
#     --test_csv "test.csv" \
#     --data_root "PfadZuNii" \
#     --model_path "best_base_model.pt" \
#     --device cuda \
#     --distance_metric euclidean \
#     --K 5
