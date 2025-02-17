import torch
import torch.nn as nn
import logging

def parse_combo_str_to_vec(combo_str):
    return [int(x) for x in combo_str.split('-')]

class MultiLabelHead(nn.Module):
    """
    3-Klassen-Kopf für Multi-Label
    """
    def __init__(self, in_dim=512, num_labels=3):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_labels)
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, emb):
        """
        emb: Tensor (B, in_dim)
        Returns: logits (B, num_labels)
        """
        logits = self.classifier(emb)
        return logits

    def compute_multilabel_loss(self, logits, labels):
        """
        logits: (B, num_labels)
        labels: (B, num_labels), float32 {0,1}
        => BCEWithLogitsLoss
        """
        return self.bce_loss_fn(logits, labels)


def compute_multilabel_metrics(
    y_true,   # Liste von [0,1,1] 
    y_scores, # Liste der vorhergesagten float scores => shape (3,) pro Patient
    threshold=0.5
):
    """
    y_true: List[List[int]], z. B. [[0,1,1],[1,0,1],...]
    y_scores: List[List[float]], Sigmoid-Output => shape [N,3]
    threshold: 0.5 => predicted=1 if score>=0.5 else 0

    Returns: dict mit f1 fibrose/emphysem/nodule + macro_f1
    """
    import numpy as np

    # 3 Labels
    TP = [0,0,0]
    FP = [0,0,0]
    FN = [0,0,0]

    for gt_vec, score_vec in zip(y_true, y_scores):
        # threshold
        pred_vec = [1 if s>=threshold else 0 for s in score_vec]
        for i in range(3):
            if gt_vec[i]==1 and pred_vec[i]==1:
                TP[i]+=1
            elif gt_vec[i]==0 and pred_vec[i]==1:
                FP[i]+=1
            elif gt_vec[i]==1 and pred_vec[i]==0:
                FN[i]+=1

    results = {}
    sum_f1 = 0
    label_names = ["fibrose","emphysem","nodule"]
    for i, name in enumerate(label_names):
        prec = TP[i]/(TP[i]+FP[i]) if (TP[i]+FP[i])>0 else 0
        rec  = TP[i]/(TP[i]+FN[i]) if (TP[i]+FN[i])>0 else 0
        f1_i = 0
        if prec+rec>0:
            f1_i = 2*prec*rec/(prec+rec)

        results[f"{name}_precision"] = prec
        results[f"{name}_recall"]    = rec
        results[f"{name}_f1"]        = f1_i
        sum_f1 += f1_i

    results["macro_f1"] = sum_f1/3.0
    return results

def evaluate_multilabel_classification(
    trainer,
    df,
    threshold=0.5
):
    """
    pro Patient => Embedding => Logits => Sigmoid => predict.
    """
    # 3-Label-Kopf
    trainer.base_cnn.eval()
    trainer.mil_agg.eval()
    trainer.classifier.eval()

    y_true = []
    y_scores = []

    for idx, row in df.iterrows():
        pid = row["pid"]
        study_yr = row["study_yr"]
        combo_str = row["combination"]

        # ground truth
        gt = parse_combo_str_to_vec(combo_str)

        with torch.no_grad():
            emb = trainer.compute_patient_embedding(pid, study_yr)
            logits = trainer.classifier(emb)  # shape (1,3)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        y_true.append(gt)
        y_scores.append(probs.tolist())

    return compute_multilabel_metrics(y_true, y_scores, threshold=threshold)
