import os
import sys
import torch
import pandas as pd

# from training.data_loader import SinglePatientDataset

################################################################################
# 1) compute_embeddings
################################################################################
def compute_embeddings(
    trainer,
    df,
    data_root,
    device='cuda'
):
    """
    Berechnet die Embeddings für jeden Patienten in df (Spalten: [pid, study_yr, combination]).
    Nutzung: trainer.compute_patient_embedding(...)
    Returns ein Dict: embeddings[(pid, study_yr)] = (emb_tensor, combination)
    """
    trainer.base_cnn.eval()
    trainer.mil_agg.eval()
    trainer.device = device

    embeddings = {}
    for idx, row in df.iterrows():
        pid = row['pid']
        study_yr = row['study_yr']
        combo = row['combination']

        with torch.no_grad():
            emb = trainer.compute_patient_embedding(pid, study_yr)
            emb = emb.squeeze(0).cpu()  # (d,)
        embeddings[(pid, study_yr)] = (emb, combo)
    return embeddings


################################################################################
# 2) compute_precision_recall_map
################################################################################
def compute_precision_recall_map(
    embeddings,
    K=10,
    distance_metric='euclidean'
):
    """
    embeddings: dict[(pid,study_yr)] = (tensor(d,), combo_str)
    Gibt Dictionary: {'precision@K':..., 'recall@K':..., 'mAP':...}
    """
    keys = list(embeddings.keys())
    n = len(keys)

    emb_list = []
    combo_list = []
    for k in keys:
        emb_list.append(embeddings[k][0])
        combo_list.append(embeddings[k][1])

    emb_tensor = torch.stack(emb_list, dim=0)  # (n,d)

    if distance_metric == 'euclidean':
        dist_matrix = torch.cdist(
            emb_tensor.unsqueeze(0),
            emb_tensor.unsqueeze(0),
            p=2
        ).squeeze(0)  # (n,n)
    elif distance_metric == 'cosine':
        sim = emb_tensor @ emb_tensor.t()
        norm = emb_tensor.norm(dim=1, keepdim=True)
        denom = norm @ norm.t()
        cos_sim = sim / denom
        dist_matrix = 1 - cos_sim
    else:
        raise ValueError(f"Unknown distance metric {distance_metric}")

    precision_sum = 0.0
    recall_sum    = 0.0
    AP_sum        = 0.0

    for i in range(n):
        combo_i = combo_list[i]
        distances_i = dist_matrix[i]
        sorted_indices = torch.argsort(distances_i, dim=0)

        # Ranking ohne i
        rank = [idx_j.item() for idx_j in sorted_indices if idx_j!=i]

        # Relevants = alle mit combo == combo_i (außer i)
        relevant_indices = [j for j,c in enumerate(combo_list) if (c==combo_i and j!=i)]
        R = len(relevant_indices)

        # Precision@K
        topK = rank[:K]
        num_correct = sum(combo_list[j] == combo_i for j in topK)
        precisionK_i = num_correct / K

        # Recall@K
        recallK_i = (num_correct / R) if R>0 else 0.0

        precision_sum += precisionK_i
        recall_sum    += recallK_i

        # AP-Berechnung
        found = 0
        AP_i  = 0.0
        for r, j_idx in enumerate(rank):
            if combo_list[j_idx] == combo_i:
                found += 1
                prec_r = found / (r+1)
                AP_i += prec_r
        if R>0:
            AP_i /= R
        AP_sum += AP_i

    precisionK = precision_sum / n
    recallK    = recall_sum / n
    mAP        = AP_sum / n

    result = {
        "precision@K": precisionK,
        "recall@K": recallK,
        "mAP": mAP
    }
    return result


################################################################################
# 3) evaluate_model
################################################################################
def evaluate_model(
    trainer,
    data_csv,
    data_root,
    K=5,
    distance_metric='euclidean',
    device='cuda'
):
    """
    Wrapper, der df einliest, Embeddings berechnet, compute_precision_recall_map aufruft
    """
    df = pd.read_csv(data_csv)
    emb_dict = compute_embeddings(trainer, df, data_root, device=device)
    metrics = compute_precision_recall_map(emb_dict, K=K, distance_metric=distance_metric)

    print(f"[evaluate_model] K={K}, metric={distance_metric}:")
    print(f" Precision@K = {metrics['precision@K']:.4f}")
    print(f" Recall@K    = {metrics['recall@K']:.4f}")
    print(f" mAP         = {metrics['mAP']:.4f}")
    return metrics
