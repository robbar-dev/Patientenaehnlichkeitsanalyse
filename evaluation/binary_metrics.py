import torch
import numpy as np
import logging

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

def evaluate_binary_classification(
    trainer,
    df,
    data_root,
    threshold=0.5,
    device='cuda',
    do_plot_roc=False,
    roc_plot_path='roc_curve.png',
    do_plot_cm=False,
    cm_plot_path='confusion_matrix.png'
):
    """
    Führt eine binäre Evaluierung durch, basierend auf 
    trainer.compute_patient_embedding(pid,study_yr).
    Annahme: 'combination' in df ist '1-0-0' = 1 (krank) 
             und '0-0-1' = 0 (gesund).

    Returns:
      metrics: dict => {'acc': float, 'auc': float, 'confusion_matrix': np.array(...), ...}
    """
    trainer.base_cnn.eval()
    trainer.mil_agg.eval()
    # Falls du classifier hast, z. B. in Binärfall => trainer.classifier.eval()
    # (Der Trainer muss natürlich binäres Labeling + BCE haben.)

    y_true = []
    y_scores = []  # Probability p(=1)
    
    # 1) Sammeln von Label + Score
    for i, row in df.iterrows():
        pid = row['pid']
        study_yr = row['study_yr']
        combo_str = row['combination']

        # MAPPEN => 1-0-0 => 1,  0-0-1 => 0
        label = 1 if combo_str.startswith("1-0-0") else 0  # oder was immer du im Datensatz hast

        with torch.no_grad():
            emb = trainer.compute_patient_embedding(pid, study_yr)
            # => shape (1,512)
            # optional: logits = trainer.classifier(emb)
            logits = trainer.classifier(emb)  # shape (1,1) im Binärfall oder (1,2) => up to you

            prob = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            # Falls shape(1,) => prob[0], falls shape(2,) => prob[1], ...
            # Hier => shape (1,) => prob = prob[0]
            # -> Du kannst anpassen je nach Setup. 
            score_1 = prob[0]  # Probability für Klasse=1

        y_true.append(label)
        y_scores.append(score_1)

    # 2) Metriken:
    # Accuracy
    preds = [1 if s >= threshold else 0 for s in y_scores]
    correct = sum(1 for p,gt in zip(preds,y_true) if p==gt)
    acc = correct/len(y_true) if len(y_true)>0 else 0.0

    # AUC
    unique_labels = set(y_true)
    auc_val = 0.0
    if len(unique_labels) > 1:  # damit roc_auc_score nicht crasht
        auc_val = roc_auc_score(y_true, y_scores)

    # Confusion Matrix
    cm = confusion_matrix(y_true, preds)  # shape(2,2)

    # Logging
    logging.info(f"[BinaryMetrics] ACC={acc:.4f}, AUC={auc_val:.4f}, threshold={threshold}")

    # => Plot ROC ?
    if do_plot_roc and len(unique_labels)>1:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plot_roc_curve(fpr, tpr, auc_val, roc_plot_path)

    # => Plot ConfusionMatrix ?
    if do_plot_cm:
        plot_confusion_matrix(cm, cm_plot_path)

    metrics = {
        'acc': acc,
        'auc': auc_val,
        'confusion_matrix': cm
    }
    return metrics


def plot_roc_curve(fpr, tpr, auc_val, out_path="roc_curve.png"):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr, tpr, color='red', label=f"ROC curve (AUC={auc_val:.4f})")
    plt.plot([0,1],[0,1], color='blue', linestyle='--', label="Chance line")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc='best')
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info(f"Saved ROC curve => {out_path}")

def plot_confusion_matrix(cm, out_path="confusion_matrix.png"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    
    # Labels in Zellen
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j, i, cm[i,j], ha="center", va="center", color="black")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info(f"Saved ConfusionMatrix => {out_path}")
