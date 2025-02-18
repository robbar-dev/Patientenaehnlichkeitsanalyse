import argparse
import os
import sys
import logging
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training.tester import BinaryTripletTester
from evaluation.metrics import compute_embeddings, compute_precision_recall_map
from evaluation.binary_metrics import evaluate_binary_classification

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test-Skript: IR-Metriken + binäre Metriken (ACC, AUC, CM, ROC) + Plots + CSV + TSNE."
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
                        help="K für Precision@K, Recall@K (IR-Metriken).")
    parser.add_argument("--distance_metric", type=str, default="euclidean",
                        choices=["euclidean","cosine"],
                        help="Distanzmetrik für IR-Metriken.")
    parser.add_argument("--log_file", type=str, default="test.log",
                        help="Logging-Datei.")
    parser.add_argument("--output_dir", type=str, default="test_plots",
                        help="In diesem Verzeichnis werden Plots & CSV gespeichert.")
    parser.add_argument("--result_csv", type=str, default="results_test_data.csv",
                        help="Name der CSV-Datei für die Testergebnisse.")

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

def parse_combo_str_to_label_binary(combo_str):
    """
    Wieder combo Umschreibung
    """
    if combo_str.startswith("1-0-0"):
        return 1
    else:
        return 0

def plot_roc_curve(fpr, tpr, auc_val, out_path="roc_curve.png"):
    """
    Zeichnet eine ROC-Kurve
    """
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    plt.plot(fpr, tpr, color='red', label=f"ROC curve (AUC={auc_val:.4f})")
    plt.plot([0,1],[0,1], color='blue', linestyle='--', label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.xlim(0,1)     
    plt.ylim(0,1)      
    plt.legend(loc='best')
    plt.tight_layout() 
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info(f"Saved ROC curve (no margin) => {out_path}")

def plot_test_embeddings(tester, df_test, data_root, device, output_dir):
    """
    2D-t-SNE-Darstellung des Test-Embedding-Raums.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    embeddings_list = []
    labels_list = []  # 0 / 1

    for _, row in df_test.iterrows():
        pid = row['pid']
        study_yr = row['study_yr']
        combo_str = row['combination']
        label_bin = parse_combo_str_to_label_binary(combo_str)  # 0 oder 1

        emb = tester.compute_patient_embedding(pid, study_yr)  # => (1,512)
        emb_np = emb.squeeze(0).detach().cpu().numpy()         # => (512,)
        embeddings_list.append(emb_np)
        labels_list.append(label_bin)

    if len(embeddings_list) == 0:
        logging.warning("Keine Embeddings gefunden => t-SNE-Plots übersprungen.")
        return

    embeddings_arr = torch.tensor(embeddings_list)
    embeddings_arr = embeddings_arr.numpy()  # => (N, 512)

    # t-SNE
    projector = TSNE(n_components=2, random_state=42)
    coords_2d = projector.fit_transform(embeddings_arr)  # => (N,2)

    # Plot
    plt.figure(figsize=(6,5))
    coords_normal = coords_2d[[i for i,l in enumerate(labels_list) if l==0]]
    coords_abnorm = coords_2d[[i for i,l in enumerate(labels_list) if l==1]]

    plt.scatter(coords_normal[:,0], coords_normal[:,1],
                 label='Normal', alpha=0.6)
    plt.scatter(coords_abnorm[:,0], coords_abnorm[:,1],
                 label='Abnormal', alpha=0.6)

    plt.title("t-SNE Test Embeddings")
    plt.legend(loc='best')
    outpath = os.path.join(output_dir, "test_embeddings_tsne.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    logging.info(f"Test-Embeddings (t-SNE) gespeichert => {outpath}")

def plot_ir_metrics(precision, recall, mAP, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(6,4))
    metrics = ['Precision@K', 'Recall@K', 'mAP']
    values  = [precision,     recall,     mAP]
    plt.bar(metrics, values, color=['blue','green','red'])
    plt.ylim(0, 1.0)
    plt.title("IR-Metrics auf Testset")
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
    outpath = os.path.join(output_dir, "ir_metrics.png")
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"IR-Metriken Plot gespeichert => {outpath}")

def plot_binary_metrics(acc, auc_val, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(5,4))
    metrics = ['ACC', 'AUC']
    values  = [acc,    auc_val]
    plt.bar(metrics, values, color=['purple','orange'])
    plt.ylim(0, 1.0)
    plt.title("Binäre Metriken auf Testset")
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
    outpath = os.path.join(output_dir, "binary_metrics.png")
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Binäre Metriken Plot gespeichert => {outpath}")

def save_results_csv(
    precision, recall, mAP,
    acc, auc_val,
    tp, tn, fp, fn,
    csv_path
):
    import csv
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        if not file_exists:
            header = [
                "Precision@K","Recall@K","mAP",
                "ACC","AUC",
                "TP","TN","FP","FN"
            ]
            writer.writerow(header)
        row = [
            f"{precision:.4f}", f"{recall:.4f}", f"{mAP:.4f}",
            f"{acc:.4f}", f"{auc_val:.4f}",
            tp, tn, fp, fn
        ]
        writer.writerow(row)
    logging.info(f"Testergebnisse in {csv_path} gespeichert.")

def main():
    args = parse_args()
    setup_logging(args.log_file)

    logging.info("=== TEST / INFERENCE ===")
    logging.info(f"Test-CSV={args.test_csv}, data_root={args.data_root}, model_path={args.model_path}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Checkpoint {args.model_path} existiert nicht.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    df_test = pd.read_csv(args.test_csv)

    # Tester anlegen & Checkpoint laden
    tester = BinaryTripletTester(
        data_root=args.data_root,
        device=args.device,
        model_name='resnet18',
        freeze_blocks=[0,1],
        aggregator_name='mil',
        agg_hidden_dim=128,
        agg_dropout=0.2,
        roi_size=(96,96,3),
        overlap=(10,10,1)
    )
    tester.load_checkpoint(args.model_path)

    # IR-Metriken
    emb_dict = compute_embeddings(
        trainer=tester,
        df=df_test,
        data_root=args.data_root,
        device=args.device
    )
    test_metrics = compute_precision_recall_map(
        embeddings=emb_dict,
        K=args.K,
        distance_metric=args.distance_metric
    )
    precK = test_metrics['precision@K']
    recK  = test_metrics['recall@K']
    mapv  = test_metrics['mAP']
    logging.info(f"[IR] => Precision@K={precK:.4f}, Recall@K={recK:.4f}, mAP={mapv:.4f}")
    plot_ir_metrics(precK, recK, mapv, args.output_dir)

    roc_path = os.path.join(args.output_dir, "roc_curve.png")
    cm_path  = os.path.join(args.output_dir, "confusion_matrix.png")
    
    bin_res = evaluate_binary_classification(
        trainer=tester,
        df=df_test,
        data_root=args.data_root,
        device=args.device,
        threshold=0.5,
        do_plot_roc=False,  # manuelles plotten besser
        do_plot_cm=True,
        cm_plot_path=cm_path
    )

    y_true = []
    y_scores = []
    with torch.no_grad():
        for _, row in df_test.iterrows():
            pid = row['pid']
            study_yr = row['study_yr']
            label = parse_combo_str_to_label_binary(row['combination'])
            logits = tester.compute_patient_logits(pid, study_yr)  # shape (1,1)
            prob   = torch.sigmoid(logits).item()
            y_true.append(label)
            y_scores.append(prob)
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_val_manual = roc_auc_score(y_true, y_scores)

    plot_roc_curve(fpr, tpr, auc_val_manual, out_path=roc_path)
    
    acc_val = bin_res["acc"]
    auc_val = auc_val_manual
    cm      = bin_res["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()
    logging.info(f"[Binary] => ACC={acc_val:.4f}, AUC={auc_val:.4f}, "
                 f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    plot_binary_metrics(acc_val, auc_val, args.output_dir)

    csv_fullpath = os.path.join(args.output_dir, args.result_csv)
    save_results_csv(
        precision=precK,
        recall=recK,
        mAP=mapv,
        acc=acc_val,
        auc_val=auc_val,
        tp=tp, tn=tn, fp=fp, fn=fn,
        csv_path=csv_fullpath
    )

    # t-SNE-Plot der Test-Embeddings
    plot_test_embeddings(tester, df_test, args.data_root, args.device, args.output_dir)

    logging.info("=== TEST DONE ===")

if __name__=="__main__":
    main()


# python3.11 training\test_model.py `
#     --test_csv "C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V7\test\nlst_subset_v7_test.csv" `
#     --data_root "D:\thesis_robert\NLST_subset_v7" `
#     --model_path best_model.pt `
#     --device cuda `
#     --distance_metric euclidean `
#     --K 3 `
#     --output_dir plots `
#     --result_csv results_test_data.csv

