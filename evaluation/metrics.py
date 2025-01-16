import os
import sys
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

# 1) Projektpfad
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training.data_loader import SinglePatientDataset

#from training.trainer import TripletTrainer

################################################################################
# 1) compute_embeddings: Für jeden Patienten in df -> CNN+MIL -> embedding
################################################################################
def compute_embeddings(
    trainer,
    df,
    data_root,
    device='cuda',
    batch_size=16
):
    """
    Berechnet die Embeddings für jeden Patienten in df (Spalten: pid, study_yr, combination).
    trainer: dein TripletTrainer-Objekt, das u.a. compute_patient_embedding(...) besitzt
    df: DataFrame mit Spalten ['pid','study_yr','combination']
    data_root: Pfad zu den .nii.gz, falls trainer.compute_patient_embedding(...) das braucht
    device: 'cuda' oder 'cpu'
    batch_size: falls Du eine separate Embedding-Batchsize brauchst (bei dir aber i.d.R. fix)
    
    Returns:
      embeddings: dict[ (pid,study_yr) ] = (embedding_tensor, combination)
                  embedding_tensor: shape (d,) oder (1,d)
                  combination: string
    """
    # Wichtig: Hier der Trainer, der compute_patient_embedding(...) bereitstellt
    trainer.device = device  # nur als Sicherheit
    trainer.base_cnn.eval()
    trainer.mil_agg.eval()

    embeddings = {}

    # Durchlaufe DataFrame
    for idx, row in df.iterrows():
        pid = row['pid']
        study_yr = row['study_yr']
        combination = row['combination']

        with torch.no_grad():
            emb = trainer.compute_patient_embedding(pid, study_yr)
            # emb shape: (1, d)
            # Squeeze, falls du (1,d) -> (d)
            emb = emb.squeeze(0).to('cpu')
        
        embeddings[(pid, study_yr)] = (emb, combination)
    return embeddings


################################################################################
# 2) compute_precision_recall_map: Berechnet Precision@K, Recall@K, mAP
################################################################################
def compute_precision_recall_map(
    embeddings,
    K=10,
    distance_metric='euclidean'
):
    """
    embeddings: dict[ (pid, study_yr) ] = (embedding (d,), combination)
    K: int, z.B. 5
    distance_metric: 'euclidean' oder 'cosine'
    
    Returns:
      result_dict = {
        'precision@K': float,
        'recall@K': float,
        'mAP': float
      }
    """

    # 1) Hole alle Keys, Embeddings, Combos in Listen
    keys = list(embeddings.keys())
    n = len(keys)

    emb_list = []
    combo_list = []
    for k in keys:
        emb_list.append(embeddings[k][0])     # (d,)
        combo_list.append(embeddings[k][1])  # string
    
    # 2) Erzeuge Tensor => shape (n,d)
    emb_tensor = torch.stack(emb_list, dim=0)  # (n,d)

    # 3) Distanzmatrix (n,n)
    if distance_metric == 'euclidean':
        # torch.cdist: (n,d) -> expand -> (n,n)
        dist_matrix = torch.cdist(emb_tensor.unsqueeze(0), emb_tensor.unsqueeze(0), p=2).squeeze(0)  
    elif distance_metric == 'cosine':
        # cos_dist = 1 - cos_sim
        sim = emb_tensor @ emb_tensor.t()  # (n,n)
        norm = emb_tensor.norm(dim=1, keepdim=True)  # (n,1)
        denom = norm @ norm.t()  # (n,n)
        cos_sim = sim / denom
        dist_matrix = 1 - cos_sim
    else:
        raise ValueError(f"Unknown distance metric {distance_metric}")

    # 4) Precision@K, Recall@K, mAP => Summen
    precision_sum = 0.0
    recall_sum    = 0.0
    AP_sum        = 0.0

    for i in range(n):
        combo_i = combo_list[i]
        # Distanz zu allen
        distances_i = dist_matrix[i]  # shape (n,)
        # sort ascending
        sorted_indices = torch.argsort(distances_i, dim=0)

        # Baue Ranking ohne i selbst
        rank = []
        for idx_j in sorted_indices:
            if idx_j != i:
                rank.append(idx_j.item())  # int
                
        # relevant = alle j, combo_list[j]==combo_i, j!=i
        relevant_indices = [j for j,c in enumerate(combo_list) if (c == combo_i and j!=i)]
        R = len(relevant_indices)  # # relevants

        # Precision@K
        topK = rank[:K]
        num_correct = sum(combo_list[j] == combo_i for j in topK)
        precisionK_i = num_correct / K
        
        # Recall@K
        recallK_i = (num_correct / R) if R>0 else 0.0

        precision_sum += precisionK_i
        recall_sum    += recallK_i

        # AP => average precision
        # laeuft durch rank, addieren precision@r, wann immer j relevant.
        found = 0
        AP_i = 0.0
        for r, j_idx in enumerate(rank):
            if combo_list[j_idx] == combo_i:
                found += 1
                prec_r = found / (r+1)  # Precision at rank r
                AP_i += prec_r
        if R>0:
            AP_i /= R
        AP_sum += AP_i

    precisionK = precision_sum / n
    recallK    = recall_sum / n
    mAP        = AP_sum / n

    result_dict = {
        'precision@K': precisionK,
        'recall@K': recallK,
        'mAP': mAP
    }
    return result_dict


################################################################################
# 3) evaluate_model: Hilfsfunktion => df laden => embeddings berechnen => IR-Metriken
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
    Lädt data_csv (muss Spalten [pid,study_yr,combination] besitzen),
    berechnet Embeddings per 'compute_embeddings()',
    ruft 'compute_precision_recall_map()' auf,
    und druckt die Ergebnisse.

    Args:
      trainer: TripletTrainer (hat compute_patient_embedding())
      data_csv: Pfad zur CSV
      data_root: Pfad zu .nii / .nii.gz
      K: int, z.B. 5
      distance_metric: 'euclidean' oder 'cosine'
      device: 'cuda' oder 'cpu'
    """
    import pandas as pd
    df = pd.read_csv(data_csv)
    emb_dict = compute_embeddings(trainer, df, data_root, device=device)
    metrics = compute_precision_recall_map(emb_dict, K=K, distance_metric=distance_metric)
    print(f"Evaluation (K={K}, metric={distance_metric}):\n"
          f" Precision@K= {metrics['precision@K']:.4f}\n"
          f" Recall@K=    {metrics['recall@K']:.4f}\n"
          f" mAP=         {metrics['mAP']:.4f}")
    return metrics


################################################################################
# Beispiel main => Zum Testen / Debug (optional)
################################################################################
if __name__ == "__main__":

    data_csv  = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\validation\nlst_subset_v5_validation.csv"
    data_root = r"D:\thesis_robert\NLST_subset_v5_nifti_3mm_Voxel"

    # Erstelle / Lade dein Trainer-Modell (vllt. trainiertes)
    # => Pseudocode:
    # trainer = TripletTrainer(
    #     df=None,  # hier irrelevant
    #     data_root=data_root,
    #     device='cuda',
    #     lr=1e-4,
    #     margin=1.0
    #     # ...
    # )

    #evaluate_model(trainer, data_csv, data_root, K=5, distance_metric='euclidean', device='cuda')
