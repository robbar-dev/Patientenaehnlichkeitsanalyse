import torch
import numpy as np
import logging
import os

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

def evaluate_binary_classification(
    trainer,
    df,
    data_root,
    threshold=0.5,
    device='cuda',
    do_plot_roc=False,
    roc_plot_path='plots/roc_curve.png',
    do_plot_cm=False,
    cm_plot_path='plots/confusion_matrix.png'
):
    """
    Führt eine binäre Evaluierung durch, basierend auf
    trainer.compute_patient_embedding(pid,study_yr) + trainer.classifier(...).
    Annahme: 'combination' in df ist '1-0-0' => 1 (abnormal) und '0-0-1' => 0 (normal),
    oder Du passt die Logik an deine CSV an.

    Returns:
      metrics: dict => {
         'acc': float,
         'auc': float,
         'confusion_matrix': np.array(...),
      }
    """
    trainer.base_cnn.eval()
    trainer.mil_agg.eval()
    # Binärer Kopf => shape(1,1)
    trainer.classifier.eval()

    y_true = []
    y_scores = []  # Probability für Label=1

    for _, row in df.iterrows():
        pid = row['pid']
        study_yr = row['study_yr']
        combo_str = row['combination']

        # => Mapping: '1-0-0' => 1, '0-0-1' => 0
        # Du passt es an, falls Du noch andere Patterns hast
        if combo_str.startswith("1-0-0"):
            label = 1
        else:
            label = 0

        with torch.no_grad():
            emb = trainer.compute_patient_embedding(pid, study_yr)  # => (1,512)
            logits = trainer.classifier(emb)  # => (1,1)
            # => Sigmoid => Probability
            prob = torch.sigmoid(logits).item()  # => float
        y_true.append(label)
        y_scores.append(prob)

    # Accuracy => threshold
    preds = [1 if s>=threshold else 0 for s in y_scores]
    correct = sum(p==gt for p,gt in zip(preds,y_true))
    acc = correct/len(y_true) if len(y_true)>0 else 0.0

    # AUC
    unique_labels = set(y_true)
    if len(unique_labels)>1:
        auc_val = roc_auc_score(y_true, y_scores)
    else:
        auc_val = 0.0

    # Confusion Matrix
    cm = confusion_matrix(y_true, preds)

    logging.info(f"[BinaryEval] ACC={acc:.4f}, AUC={auc_val:.4f}")

    # Optional: ROC-Kurve + CM plotten
    if do_plot_roc and len(unique_labels)>1:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plot_roc_curve(fpr, tpr, auc_val, roc_plot_path)

    if do_plot_cm:
        plot_confusion_matrix(cm, cm_plot_path)

    return {
        'acc': acc,
        'auc': auc_val,
        'confusion_matrix': cm
    }

def plot_roc_curve(fpr, tpr, auc_val, out_path="plots/roc_curve.png"):
    import matplotlib.pyplot as plt
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    plt.plot(fpr, tpr, color='red', label=f"ROC curve (AUC={auc_val:.4f})")
    plt.plot([0,1],[0,1], color='blue', linestyle='--', label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Binary)")
    plt.legend(loc='best')
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info(f"Saved ROC curve => {out_path}")

def plot_confusion_matrix(cm, out_path="plots/confusion_matrix.png"):
    import matplotlib.pyplot as plt
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix (Binary)")
    plt.colorbar()
    tick_marks = [0,1]
    plt.xticks(tick_marks, ["Normal","Abnormal"])
    plt.yticks(tick_marks, ["Normal","Abnormal"])
    plt.tight_layout()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center", color="black")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info(f"Saved confusion matrix => {out_path}")
